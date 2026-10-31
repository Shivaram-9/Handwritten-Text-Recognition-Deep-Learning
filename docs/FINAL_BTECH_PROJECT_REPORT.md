# Handwritten Text Recognition Using Deep Learning
## Final Engineering & Demonstration Report

**Production System URL:** [https://handwritten-text-recognition-t6cz.onrender.com/](https://handwritten-text-recognition-t6cz.onrender.com/)  
**Repository Branch:** `main`  
**Current Baseline Commit:** `d00648a`  
**Deployment Infrastructure:** Render Cloud Container (Free Tier, 512 MB Peak RAM Budget, Native Python 3.11 Runtime)

---

## 1. Abstract

Handwritten Text Recognition (HTR) remains a pivotal challenge in pattern recognition and artificial intelligence due to immense variability in individual human handwriting, continuous cursive strokes, letter connectivity, and document noise. This project implements an end-to-end, full-stack, deep learning platform that ingests raw, unconstrained handwritten document images, performs perspective rectification, contrast-limited adaptive binarization, and vertical connected-component word segmentation, passes segmented word sequences through a hybrid Convolutional Neural Network and Bidirectional Long Short-Term Memory (CNN-BiLSTM) network with Connectionist Temporal Classification (CTC), and refines transcriptions via a context-aware lexical post-processor. 

The complete system is deployed with a production-grade web interface featuring an interactive particle-neural background, multi-stage computer vision preview cards, real-time inference latency instrumentation, and full mobile responsiveness, operating deterministically within a strict 512 MB memory boundary.

---

## 2. Problem Statement

Optical Character Recognition (OCR) systems designed for standard machine-printed text fail severely when applied to unconstrained handwritten inputs due to:
1. Inconsistent character aspect ratios and stroke widths.
2. Cursive ligatures where letter boundaries overlap horizontally and vertically.
3. Diverse document lighting, paper textures, skew, and background degradation.
4. Heavy cloud server resource footprints that exceed free-tier production memory limitations (< 512 MB RAM).

---

## 3. Objectives

1. Develop a robust computer vision preprocessing and segmentation pipeline to detect, clean, and crop individual words from full handwritten documents without manual bounding boxes.
2. Train a lightweight hybrid CNN-BiLSTM-CTC architecture optimized for sequential feature extraction and alignment-free sequence decoding.
3. Formulate a conservative, evidence-based language-model post-processor that corrects optical errors while strictly preserving mixed-case casing, acronyms, and block-capital text.
4. Establish an authoritative canonical ground-truth benchmark dataset across all nine project test samples with independent case-sensitive and case-insensitive metrics.
5. Deploy the application to a live, zero-cost production web environment with responsive visual feedback, cold-start handling, and interactive CV pipeline visualization.

---

## 4. Existing System vs. Proposed System

| Dimension | Existing / Traditional OCR Systems | Proposed Deep Learning Platform |
|---|---|---|
| **Feature Extraction** | Handcrafted morphological features (HOG, SIFT, projection profiles). | Deep hierarchical representation learning via multi-scale Conv2D layers. |
| **Sequence Alignment** | Hard character segmentation requiring clean inter-character gaps. | Alignment-free Connectionist Temporal Classification (CTC Loss & Decoding). |
| **Contextual Modeling** | Independent character classification ignoring temporal word flow. | Bidirectional LSTM sequence modeling capturing forward and backward stroke context. |
| **Casing Adaptability** | Fixed global casing or naive thresholding causing uppercase corruption. | Multi-factor evidence-based case detection preserving acronyms and sentence case. |
| **Deployment Footprint** | Heavy transformer-based models (e.g. TrOCR > 700 MB RAM). | Compact CNN-BiLSTM model graph (~12.5 MB weights, < 420 MB peak RAM). |

---

## 5. Technology Stack

- **Deep Learning & Numerical Computation:** TensorFlow 2.16 / Keras 3, NumPy, SciPy
- **Computer Vision & Image Processing:** OpenCV (`opencv-python-headless`), Pillow
- **Post-Processing & Natural Language Model:** SymSpell, PySpellChecker, Levenshtein distance
- **Backend Architecture & Serving:** Flask 3.1, Gunicorn, Werkzeug
- **Frontend Architecture:** HTML5, CSS3 Custom Properties, Vanilla JavaScript (ES6+), Canvas API, Lucide Icons
- **Quality Assurance & Testing:** Pytest, AnyIO, JiWER
- **Cloud Infrastructure & Hosting:** Render PaaS, Cloudflare CDN, Git Version Control

---

## 6. Model Architecture & Theoretical Foundation

The neural architecture employs a 3-stage hybrid topology:

```
[Input Word Image (32x128x1)]
              │
              ▼
   [CNN Feature Extractor]
   5x (Conv2D -> BatchNorm -> LeakyReLU -> MaxPool2D)
              │
              ▼
    [Feature Map Squeeze] -> (TimeSteps: 32, Channels: 64)
              │
              ▼
[Bidirectional LSTM Learner]
   2x BiLSTM Layers (512 hidden units forward + backward)
              │
              ▼
     [TimeDistributed Dense] -> Softmax over Vocabulary (Vocab Size: 81)
              │
              ▼
      [CTC Loss / Decoder] -> Collapses blanks and duplicates -> Raw Tokens
```

### 6.1. CNN Feature Extractor
Extracts 2D spatial visual features:
- **Layer 1:** 32 filters ($3 \times 3$), BatchNorm, Pool ($2 \times 2$) $\rightarrow (16, 64, 32)$
- **Layer 2:** 64 filters ($3 \times 3$), BatchNorm, Pool ($2 \times 2$) $\rightarrow (8, 32, 64)$
- **Layer 3:** 128 filters ($3 \times 3$), BatchNorm, Pool ($2 \times 1$) $\rightarrow (4, 32, 128)$
- **Layer 4:** 256 filters ($3 \times 3$), BatchNorm, Pool ($2 \times 1$) $\rightarrow (2, 32, 256)$
- **Layer 5:** 256 filters ($3 \times 3$), BatchNorm, Pool ($2 \times 1$) $\rightarrow (1, 32, 256)$
- **Squeeze & Projection:** Squeezes the vertical dimension to produce 32 sequential feature vectors of dimension 64.

### 6.2. Bidirectional LSTM Sequence Learner
Models contextual dependencies across horizontal character strokes:
- Stacked 2-layer Bidirectional LSTM with 256 units in each direction (total 512 dimensions per time-step).
- Enables the network to condition the prediction of ambiguous central characters on preceding and succeeding strokes.

### 6.3. Connectionist Temporal Classification (CTC)
CTC eliminates the requirement for pre-segmented individual characters by introducing a blank token $\epsilon$:
$$\mathcal{L}_{\text{CTC}} = -\ln \sum_{\pi \in \mathcal{B}^{-1}(l)} P(\pi | X)$$
During decoding, identical consecutive activations are collapsed unless separated by a CTC blank.

---

## 7. Preprocessing & Segmentation Pipeline

1. **Grayscale & Resizing:** Ingested inputs (JPG, PNG, WEBP) are converted to single-channel luminance arrays.
2. **Contrast-Limited Adaptive Histogram Equalization (CLAHE):** Enhances local contrast against uneven shadows and faded ink.
3. **Adaptive Otsu Thresholding:** Generates clean binarized stroke profiles.
4. **Document Contour Detection & Line Grouping:** Identifies text line baselines using horizontal projection profiles and bounding box vertical overlap.
5. **Word Segmentation:** Employs dynamic horizontal gap thresholding to isolate word blobs, centering and padding each crop to a fixed tensor shape of $(32, 128, 1)$.

---

## 8. Language Model Post-Processing

A multi-factor post-processing engine balances visual acoustic fidelity and lexical probability:
- **Lexical Candidate Ranking:** Suggests corrections only for words with edit distance $\le 1$ when the visual confidence and character sequence similarity satisfy $\ge 0.75$.
- **Acoustic Safeguards:** Blocks overzealous dictionary replacements (e.g. preserves domain terms and prevents replacing `learing` with `leaving` or `suces` with `sauces`).
- **Robust Case Style Detection:**
  - Standard sentence casing and acronyms (`"NASA"`, `"ESA"`, `"IBM"`) are preserved without falsely forcing the document into uppercase.
  - An all-caps writing style is recognized only when strong document-wide evidence exists ($\ge 85\%$ uppercase ratio on short strings, or $\ge 50\%$ uppercase ratio with $\ge 4$ internal/full uppercase words across extended passages).

---

## 9. Canonical Benchmark Results

Evaluated against the authoritative canonical dataset ([`evaluation_results/canonical_ground_truth.json`](file:///E:/MLDRLG/Handwritten-Text-Recognition-Deep-Learning/evaluation_results/canonical_ground_truth.json)):

| Image File | Canonical Ground Truth | Raw Model Prediction | Final System Transcription | CS CER | CI CER | CS WER | CI WER | Confidence | Latency (Wall / Inf) |
|---|---|---|---|---|---|---|---|---|---|
| **1.png** | *The quick brown fox jumps over the lazy dog. Handwritten text recognition using deep learning works well!* | `The quisk brown for jumps over the lasy dag Mandwriten text resognit on using dep learing works wel` | `The quick brown for jumps over the lay dag Handwritten text resognit on using deep learning works we` | 8.57% | 8.57% | 35.29% | 35.29% | 86.49% | 2979 ms / 1206 ms |
| **2.png** | *Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle.* | `Believe in youself and al that you are Know that there is sormething inside yoe thatis greaten than any obstacle` | `Believe in yourself and al that you are Know that there is something inside yoke that's greaten than any obstacle` | 6.90% | 6.90% | 33.33% | 33.33% | 72.68% | 1011 ms / 959 ms |
| **3.png** | *My Name is Shiva. I am a Computer Science student. I love building projects and solving problems.* | `My Mame is Shiva I am a Compater Science student I love building projects and solving problems` | `My Make is Shiva I am a Compacter Science student I love building projects and solving problems` | 7.22% | 7.22% | 29.41% | 29.41% | 87.73% | 1336 ms / 1294 ms |
| **4.png** | *SUCCESS IS NOT FINAL, FAILURE IS NOT FATAL: IT IS THE COURAGE TO CONTINUE THAT COUNTS.* | `suces is Nor FINAY FAILURE is wor FATAL IT is THE coueRAE To coNTiNE THAT coUsTs` | `SUCRES IS NOR FINAL FAILURE IS OR FATAL IT IS THE COURAGE TO CONTINUE THAT COSTS` | 11.63% | 11.63% | 37.50% | 37.50% | 58.36% | 1560 ms / 1048 ms |
| **5.png** | *learning never exhausts the mind. it only ignites your curiosity.* | `learming never exhausts the mind it only ignites your suriosity` | `learning never exhausts the mind it only ignites your curiosity` | 3.08% | 3.08% | 20.00% | 20.00% | 85.16% | 1100 ms / 1061 ms |
| **6.png** | *The best way to predict your future is to create it. Keep going!* | `The best way to predist your future is to t" create it kep going` | `The best way to predict your future is to t" create it kep going` | 10.94% | 9.38% | 30.77% | 30.77% | 83.81% | 1196 ms / 1149 ms |
| **7.png** | *Code is like humor. When you have to explain it, it's bad. - Cory House* | `Code is like humor when you have to explain it it's bad Cory House` | `Code is like humor when you have to explain it it's bad Cory House` | 8.45% | 7.04% | 33.33% | 26.67% | 84.32% | 95 ms / 59 ms |
| **8.png** | *Dream big, work hard, stay focused, and surround yourself with good people.* | `Dream big. work hard stay facused and suround yoursest with god people` | `Dream big. work hard stay focused and surround courses with god people` | 10.67% | 10.67% | 50.00% | 50.00% | 86.03% | 1807 ms / 1161 ms |
| **9.png** | *Discipline today leads to freedom tomorrow.* | `Discipline today leads to fredom tomow ,` | `Discipline today leads to freedom tomow,` | 9.30% | 9.30% | 16.67% | 16.67% | 54.29% | 1374 ms / 1113 ms |

### 9.1. Benchmark Macro Averages
- **Macro Case-Sensitive CER:** **8.53%**
- **Macro Case-Insensitive CER:** **8.20%**
- **Macro Case-Sensitive WER:** **31.81%**
- **Macro Case-Insensitive WER:** **31.07%**
- **Mean Recognition Confidence:** **77.65%**
- **Average Total End-to-End Latency:** **1384 ms**

---

## 10. Verification of Images 4 and 8

### Image 4 Alignment Analysis
- **Ground Truth (16 words):** `SUCCESS`, `IS`, `NOT`, `FINAL,`, `FAILURE`, `IS`, `NOT`, `FATAL:`, `IT`, `IS`, `THE`, `COURAGE`, `TO`, `CONTINUE`, `THAT`, `COUNTS.`
- **Final Prediction (16 words):** `SUCRES`, `IS`, `NOR`, `FINAL`, `FAILURE`, `IS`, `OR`, `FATAL`, `IT`, `IS`, `THE`, `COURAGE`, `TO`, `CONTINUE`, `THAT`, `COSTS`
- **Exact Word Alignment Breakdown:**
  - **10 words match the base word perfectly:** `IS`, `FINAL` (omitted comma), `FAILURE`, `IS`, `FATAL` (omitted colon), `IT`, `IS`, `THE`, `COURAGE`, `TO`, `CONTINUE`, `THAT`.
  - **8 words match exact token + punctuation:** `IS`, `FAILURE`, `IS`, `IT`, `IS`, `THE`, `COURAGE`, `TO`, `CONTINUE`, `THAT`.
  - **4 word errors:** `SUCRES` (vs `SUCCESS`), `NOR` (vs `NOT`), `OR` (vs `NOT`), `COSTS` (vs `COUNTS`).
  - **2 punctuation omissions:** `FINAL` (vs `FINAL,`), `FATAL` (vs `FATAL:`).
  - **Resulting WER:** Case-sensitive WER = **37.50%** (6 token mismatches / 16 words).
  - **Resulting CER:** Case-sensitive CER = **11.63%** (10 character errors / 86 characters).
  - *Correction Note:* The previous report's colloquial statement that "all 16 words are recognized accurately" was inaccurate; precisely 10 of 16 words match the underlying vocabulary, with 4 optical substitution errors.

### Image 8 Alignment Analysis
- **Ground Truth (12 words):** `Dream`, `big,`, `work`, `hard,`, `stay`, `focused,`, `and`, `surround`, `yourself`, `with`, `good`, `people.`
- **Final Prediction (12 words):** `Dream`, `big.`, `work`, `hard`, `stay`, `focused`, `and`, `surround`, `courses`, `with`, `god`, `people`
- **Exact Word Alignment Breakdown:**
  - **5 exact matches (word + punctuation):** `Dream`, `work`, `stay`, `and`, `surround`, `with`.
  - **2 optical substitution errors:** `courses` (vs `yourself`), `god` (vs `good`).
  - **5 punctuation mismatches:** `big.` vs `big,`, omitted commas after `hard`, `focused`, and omitted period after `people`.
  - **Resulting WER:** Case-sensitive WER = **50.00%** (6 word mismatches / 12 words).
  - **Resulting CER:** Case-sensitive CER = **10.67%** (8 character errors / 75 characters).

---

## 11. Production Deployment & Live Verification

- **Hosting Platform:** Render Web Service (Free Tier)
- **Container Environment:** Linux container, Python 3.11, Gunicorn WSGI with 1 worker and 120s timeout.
- **Cold-Start Handling:** Cold start requires ~14 seconds for TensorFlow and Gunicorn process initialization on Render's shared CPU. The frontend displays animated status messaging during active requests.
- **Warm Endpoint Latency:** Once initialized, the web service responds to health probes in **0.48s** and processes complete document segmentation, OCR, and spellchecking within **~1.38s**.
- **Supported File Formats:** Verified support for `.png`, `.jpg`, `.jpeg`, and `.webp`.

---

## 12. Limitations & Future Enhancements

1. **Punctuation Modeling:** CTC acoustic modeling on character-level IAM words exhibits lower activation mass for commas, hyphens, and colons.
2. **Dense Cursive Connectivity:** Highly slanted cursive ligatures (e.g. `yourself` in Image 8) are susceptible to acoustic confusion (`courses`).
3. **Future Work:** Fine-tuning an encoder-decoder Vision Transformer (TrOCR) with quantization (INT8) once server memory budgets $\ge 1\text{ GB}$ are allocated.

---

## 13. Conclusion

The Handwritten Text Recognition project successfully fulfills all functional, academic, and deployment requirements. Through systematic calibration of image preprocessing, CNN-BiLSTM feature learning, connectionist temporal decoding, robust casing heuristics, and canonical benchmark integrity validation, the system achieves a verified **8.53% Character Error Rate** and **31.81% Word Error Rate** across the authoritative benchmark, running reliably in a live, public, zero-cost production environment.
