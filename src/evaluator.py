import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Third party evaluation metrics
try:
    import jiwer
except ImportError:
    jiwer = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
except ImportError:
    canvas = None

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from src.predictor import HTRPredictor

# Configure Logging
log_file = os.path.join(Config.BASE_DIR, 'logs', 'evaluation.log')
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HTREvaluator:
    """
    Production-ready Evaluation Module for the HTR System.
    Calculates CER, WER, Exact Match Accuracy, and generates JSON/PDF reports.
    """
    def __init__(self):
        self.predictor = HTRPredictor()
        self.test_data_path = os.path.join(Config.DATA_DIR, 'processed', 'splits', 'test_split.json')
        self.results_dir = os.path.join(Config.BASE_DIR, 'evaluation_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _load_test_data(self):
        if not os.path.exists(self.test_data_path):
            logger.error(f"Test split not found at {self.test_data_path}")
            return []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _calculate_metrics(self, ground_truths, predictions):
        """Calculates CER, WER, and Exact Match Accuracy."""
        if not jiwer:
            logger.warning("jiwer library not installed. Skipping CER/WER calculation.")
            return 0.0, 0.0, 0.0
            
        # Exact Match Accuracy
        exact_matches = sum([1 for gt, pred in zip(ground_truths, predictions) if gt.strip() == pred.strip()])
        exact_match_acc = (exact_matches / len(ground_truths)) * 100 if ground_truths else 0.0
        
        # Word Error Rate and Character Error Rate
        # jiwer throws errors on completely empty reference strings, so we filter them out for safe calculation
        valid_gts = []
        valid_preds = []
        for gt, pred in zip(ground_truths, predictions):
            if gt.strip():
                valid_gts.append(gt.strip())
                # If pred is empty, replace with a space so jiwer can calculate insertions/deletions properly
                valid_preds.append(pred.strip() if pred.strip() else " ")
                
        if len(valid_gts) == 0:
            return exact_match_acc, 0.0, 0.0

        wer = jiwer.wer(valid_gts, valid_preds) * 100
        cer = jiwer.cer(valid_gts, valid_preds) * 100
        
        # We can define pseudo Precision/Recall/F1 at character level using CER
        # Character Accuracy = 100 - CER (bounded to 0)
        char_accuracy = max(0.0, 100.0 - cer)
        
        return exact_match_acc, wer, cer, char_accuracy

    def _generate_confusion_matrix(self, ground_truths, predictions):
        """Generates a character-level confusion matrix for top characters."""
        logger.info("Generating Character Confusion Matrix...")
        
        # We will track character to character mapping
        from collections import defaultdict
        import string
        
        cm = defaultdict(lambda: defaultdict(int))
        valid_chars = set(string.ascii_letters + string.digits)
        
        for gt, pred in zip(ground_truths, predictions):
            # Pad the shorter string to zip properly for basic 1-to-1 mapping
            # Note: For rigorous alignment, Levenshtein alignment is preferred, 
            # but simple zip works for a high-level overview.
            length = min(len(gt), len(pred))
            for i in range(length):
                c_gt = gt[i]
                c_pred = pred[i]
                if c_gt in valid_chars and c_pred in valid_chars:
                    cm[c_gt][c_pred] += 1
                    
        if not cm:
            logger.warning("Not enough valid characters to generate confusion matrix.")
            return None
            
        # Get top 20 most frequent ground truth characters to keep matrix readable
        char_counts = {c: sum(cm[c].values()) for c in cm.keys()}
        top_chars = sorted(char_counts, key=char_counts.get, reverse=True)[:20]
        
        matrix = np.zeros((len(top_chars), len(top_chars)))
        for i, c_gt in enumerate(top_chars):
            for j, c_pred in enumerate(top_chars):
                matrix[i, j] = cm[c_gt][c_pred]
                
        # Normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums!=0)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=False, cmap='Blues', xticklabels=top_chars, yticklabels=top_chars)
        plt.title('Character Confusion Matrix (Normalized Top 20)')
        plt.xlabel('Predicted Character')
        plt.ylabel('True Character')
        
        cm_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        
        return cm_path
        
    def _save_json_report(self, results_dict):
        json_path = os.path.join(self.results_dir, 'evaluation_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=4)
        logger.info(f"JSON Report saved to {json_path}")
        
    def _generate_pdf_report(self, metrics, cm_path):
        if not canvas:
            logger.warning("ReportLab not installed. Skipping PDF generation.")
            return
            
        pdf_path = os.path.join(self.results_dir, 'evaluation_report.pdf')
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "HTR Evaluation Report")
        
        # Metrics
        c.setFont("Helvetica", 12)
        y = height - 100
        for key, val in metrics.items():
            if isinstance(val, float):
                text = f"{key}: {val:.2f}%"
            else:
                text = f"{key}: {val}"
            c.drawString(50, y, text)
            y -= 20
            
        # Draw Image
        if cm_path and os.path.exists(cm_path):
            c.drawString(50, y - 20, "Confusion Matrix:")
            # Draw image scaled down
            c.drawImage(cm_path, 50, y - 350, width=400, height=300, preserveAspectRatio=True)
            
        c.save()
        logger.info(f"PDF Report saved to {pdf_path}")

    def evaluate(self, sample_size=None):
        """
        Runs the full evaluation pipeline on the test dataset.
        :param sample_size: Limit evaluation to N samples for speed testing.
        """
        logger.info("Starting HTR Evaluation Pipeline...")
        test_data = self._load_test_data()
        
        if not test_data:
            logger.error("No test data available for evaluation.")
            return
            
        if sample_size:
            test_data = test_data[:sample_size]
            
        total_samples = len(test_data)
        logger.info(f"Evaluating on {total_samples} samples.")
        
        ground_truths = []
        predictions = []
        confidences = []
        
        # Note: In a true production environment, predictions would be batched.
        # For simplicity and code clarity, we iterate.
        for item in tqdm(test_data, desc="Evaluating"):
            img_rel_path = item.get("image")
            gt_text = item.get("label")
            
            img_full_path = os.path.join(Config.DATA_DIR, 'processed', 'IAM', 'words', img_rel_path)
            
            if not os.path.exists(img_full_path):
                # Check raw as fallback if processing is skipped in tests
                img_full_path = os.path.join(Config.DATA_DIR, 'raw', 'IAM', 'words', img_rel_path)
            
            if os.path.exists(img_full_path):
                pred_text, conf = self.predictor.predict_image(img_full_path)
                
                # Incase the predictor fails (e.g. empty weights), it returns None
                pred_text = pred_text if pred_text is not None else ""
                
                ground_truths.append(gt_text)
                predictions.append(pred_text)
                confidences.append(conf)
            
        # Calculate Metrics
        logger.info("Calculating metrics...")
        exact_acc, wer, cer, char_acc = self._calculate_metrics(ground_truths, predictions)
        
        avg_confidence = (sum(confidences) / len(confidences)) * 100 if confidences else 0.0
        
        metrics = {
            "Total_Samples_Evaluated": total_samples,
            "Sequence_Exact_Match_Accuracy": exact_acc,
            "Character_Error_Rate_CER": cer,
            "Word_Error_Rate_WER": wer,
            "Character_Accuracy_F1_Proxy": char_acc,
            "Average_Prediction_Confidence": avg_confidence
        }
        
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"{k}: {v:.2f}%")
            else:
                logger.info(f"{k}: {v}")
                
        # Generate Confusion Matrix
        cm_path = self._generate_confusion_matrix(ground_truths, predictions)
        
        # Save Reports
        self._save_json_report(metrics)
        self._generate_pdf_report(metrics, cm_path)
        
        logger.info("Evaluation Pipeline Completed Successfully.")

if __name__ == "__main__":
    try:
        evaluator = HTREvaluator()
        # Evaluates on the first 100 samples for a quick run. Remove parameter for full evaluation.
        evaluator.evaluate(sample_size=100)
    except Exception as e:
        logger.error(f"Critical error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
