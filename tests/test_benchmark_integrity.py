import os
import json
import pytest
import numpy as np

def cer(ref, hyp):
    r_list, h_list = list(ref), list(hyp)
    d = np.zeros((len(r_list) + 1, len(h_list) + 1), dtype=np.uint16)
    for i in range(len(r_list) + 1): d[i, 0] = i
    for j in range(len(h_list) + 1): d[0, j] = j
    for i in range(1, len(r_list) + 1):
        for j in range(1, len(h_list) + 1):
            if r_list[i - 1] == h_list[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[len(r_list), len(h_list)]) / max(1, len(r_list))

def wer(ref, hyp):
    r_words = ref.split()
    h_words = hyp.split()
    d = np.zeros((len(r_words) + 1, len(h_words) + 1), dtype=np.uint16)
    for i in range(len(r_words) + 1): d[i, 0] = i
    for j in range(len(h_words) + 1): d[0, j] = j
    for i in range(1, len(r_words) + 1):
        for j in range(1, len(h_words) + 1):
            if r_words[i - 1] == h_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[len(r_words), len(h_words)]) / max(1, len(r_words))

CANONICAL_GT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "evaluation_results",
    "canonical_ground_truth.json"
)

TEST_SAMPLES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "test_samples"
)

@pytest.fixture(scope="module")
def canonical_data():
    assert os.path.exists(CANONICAL_GT_PATH), f"Missing canonical ground truth file at {CANONICAL_GT_PATH}"
    with open(CANONICAL_GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def test_benchmark_has_exact_nine_entries(canonical_data):
    samples = canonical_data.get("samples", [])
    assert len(samples) == 9, f"Expected exactly 9 benchmark samples, found {len(samples)}"

def test_no_duplicate_filenames(canonical_data):
    samples = canonical_data.get("samples", [])
    filenames = [s["filename"] for s in samples]
    assert len(filenames) == len(set(filenames)), "Duplicate filenames detected in benchmark manifest"

def test_all_benchmark_image_files_exist(canonical_data):
    samples = canonical_data.get("samples", [])
    for s in samples:
        fname = s["filename"]
        img_path = os.path.join(TEST_SAMPLES_DIR, fname)
        assert os.path.exists(img_path), f"Benchmark image file {fname} not found in {TEST_SAMPLES_DIR}"
        assert os.path.getsize(img_path) > 1000, f"Image {fname} is suspiciously empty or corrupt"

def test_ground_truth_entries_are_non_empty(canonical_data):
    samples = canonical_data.get("samples", [])
    for s in samples:
        gt = s.get("canonical_transcription", "").strip()
        assert len(gt) > 0, f"Ground truth for {s['filename']} is empty"
        assert len(s.get("visible_lines", [])) > 0, f"Visible lines missing for {s['filename']}"
        assert s.get("verification_status") == "VERIFIED_CANONICAL"

def test_image_to_reference_mapping_integrity(canonical_data):
    expected_order = [f"{i}.png" for i in range(1, 10)]
    samples = canonical_data.get("samples", [])
    actual_order = [s["filename"] for s in samples]
    assert actual_order == expected_order, f"Image mapping sequence mismatch: expected {expected_order}, got {actual_order}"

def test_separate_case_sensitive_and_insensitive_metric_calculation():
    ref = "SUCCESS IS NOT FINAL"
    hyp_lower = "success is not final"
    hyp_exact = "SUCCESS IS NOT FINAL"
    
    # Case-insensitive CER must be 0 for exact spelling regardless of casing
    ci_cer = cer(ref.lower(), hyp_lower.lower())
    assert ci_cer == 0.0
    
    # Case-sensitive CER must capture case divergence
    cs_cer = cer(ref, hyp_lower)
    assert cs_cer > 0.0
    
    # When case matches, case-sensitive CER is 0.0
    assert cer(ref, hyp_exact) == 0.0
