import os
import time
import jiwer
import json
from src.predictor import HTRPredictor

ground_truths = {
    1: "The quick brown fox jumps over the lazy dog. Handwritten text recognition using deep learning works well!",
    2: "Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle.",
    3: "My Name is Shiva. I am a Computer Science student. I love building projects and solving problems.",
    4: "SUCCESS IS NOT FINAL, FAILURE IS NOT FATAL: IT IS THE COURAGE TO CONTINUE THAT COUNTS.",
    5: "learning never exhausts the mind. it only ignites your curiosity.",
    6: "The best way to predict your future is to create it. Keep going!",
    7: "Code is like humor. When you have to explain it, it's bad. - Cory House",
    8: "Dream big, work hard, stay focused, and surround yourself with good people.",
    9: "Discipline today leads to freedom tomorrow."
}

def run_evaluation():
    predictor = HTRPredictor()

    print("=" * 80)
    print("EVALUATION OF ALL 9 TEST SAMPLES")
    print("=" * 80)

    results = []
    for i in range(1, 10):
        img_path = f"data/test_samples/{i}.png"
        gt = ground_truths[i]
        t0 = time.perf_counter()
        pred_text, conf, timings, _ = predictor.predict_image(img_path)
        t1 = time.perf_counter()
        
        pred_clean = " ".join(pred_text.split())
        
        cer = jiwer.cer(gt, pred_clean)
        wer = jiwer.wer(gt, pred_clean)
        latency = (t1 - t0) * 1000
        
        # Word breakdown
        gt_words = gt.split()
        pred_words = pred_clean.split()
        
        results.append({
            "image": f"Image {i}",
            "filename": f"{i}.png",
            "gt": gt,
            "pred": pred_clean,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "conf": round(conf, 4),
            "latency_ms": round(latency, 2),
            "exact_match": (gt.strip().lower() == pred_clean.strip().lower())
        })
        
        print(f"Image {i}:")
        print(f"  Ground Truth: {gt}")
        print(f"  Predicted   : {pred_clean}")
        print(f"  CER: {cer*100:.2f}% | WER: {wer*100:.2f}% | Conf: {conf*100:.2f}% | Latency: {latency:.1f}ms")
        print("-" * 80)

    avg_cer = sum(r["cer"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)
    print(f"AVERAGE CER: {avg_cer*100:.2f}%")
    print(f"AVERAGE WER: {avg_wer*100:.2f}%")
    
    out_dir = "evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "nine_samples_evaluation.json"), "w") as f:
        json.dump({
            "samples": results,
            "avg_cer": round(avg_cer, 4),
            "avg_wer": round(avg_wer, 4),
            "model": "best_htr_model.weights.h5"
        }, f, indent=2)
    print(f"Saved evaluation results to {os.path.join(out_dir, 'nine_samples_evaluation.json')}")

if __name__ == "__main__":
    run_evaluation()
