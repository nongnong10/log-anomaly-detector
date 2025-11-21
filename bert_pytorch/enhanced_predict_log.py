from pydantic import BaseModel
import torch
from bert_pytorch.predict_log import Predictor


# --- Updated Single-line Predictor Wrapper ---
class EnhancedPredictor(Predictor):
    """
    Extend the original Predictor to support single-log-line inference.
    If a log line is too short for the model (seq_len < expected),
    it auto-pads the sequence with itself to allow meaningful inference.
    """

    def predict_single_log(self, log_line, seq_threshold=0.5, pad_min_len=10):
        """
        Predict anomaly for a single log line, padding if too short.

        Args:
            log_line (str): Raw log line text.
            seq_threshold (float): Threshold ratio for anomaly classification.
            pad_min_len (int): Minimum pseudo-sequence length for inference.
        """
        # Convert single line into pseudo-sequence
        padded_lines = [log_line.strip()] * pad_min_len

        # Use existing multi-line prediction logic
        result = super().predict_single_log_batch(padded_lines, seq_threshold=seq_threshold)

        # Aggregate sequence results (average or majority)
        total = sum([r["total_sequences"] for r in result])
        anomalies = sum([r["anomaly_sequences"] for r in result])
        ratio = anomalies / total if total > 0 else 0.0

        is_anomaly = ratio > seq_threshold

        return {
            "input": log_line,
            "is_anomaly": is_anomaly,
            "anomalous_sequences": anomalies,
            "total_sequences": total,
            "sequence_anomaly_ratio": ratio,
        }


# --- Predictor Configuration ---
predictor_options = {
    "model_path": "./output/hdfs/bert/best_bert.pth",
    "vocab_path": "./output/hdfs/vocab.pkl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": 5,
    "adaptive_window": True,
    "seq_len": 20,
    "corpus_lines": 1,
    "on_memory": True,
    "batch_size": 32,
    "num_workers": 0,
    "num_candidates": 6,
    "output_dir": "./output/hdfs/",
    "model_dir": "./output/hdfs/bert/",
    "gaussian_mean": 0.0,
    "gaussian_std": 1.0,
    "is_logkey": True,
    "is_time": False,
    "scale_path": "./output/hdfs/scale.pkl",
    "hypersphere_loss": False,
    "hypersphere_loss_test": False,
    "test_ratio": 1,
    "mask_ratio": 0.3,
    "min_len": 2
}

predictor = EnhancedPredictor(predictor_options)


# --- Input Model ---
class LinePredictRequest(BaseModel):
    log_line: str
    seq_threshold: float = 0.5
    include_details: bool = False


# --- Prediction Function ---
def predict_line(req: LinePredictRequest):
    """
    Detect anomaly for one raw log line (auto-padded if too short).
    """
    result = predictor.predict_single_log(
        req.log_line, seq_threshold=req.seq_threshold
    )
    return {
        "input": req.log_line,
        "is_anomaly": result["is_anomaly"],
        "anomalous_sequences": result["anomalous_sequences"],
        "total_sequences": result["total_sequences"],
        "seq_threshold": req.seq_threshold,
        "sequence_anomaly_ratio": result["sequence_anomaly_ratio"],
    }


# --- Example Run ---
if __name__ == "__main__":
    # Example test log line
    sample_log = "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102"
    req = LinePredictRequest(log_line=sample_log, seq_threshold=0.5)
    result = predict_line(req)
    print(result)
