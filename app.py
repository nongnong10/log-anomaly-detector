from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model_utils import LogBERTDetector
from bert_pytorch.predict_log import Predictor  # added import

app = FastAPI(title="LogBERT Anomaly Detector")

options = {
    "model_path": "./output/hdfs/bert/best_bert.pth",
    "vocab_path": "./output/hdfs/vocab.pkl",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "window_size": 5,
    "adaptive_window": True,
    "seq_len": 20,
    "min_len": 2,
    "mask_ratio": 0.3,
    "num_candidates": 6,
    "is_logkey": True,
    "is_time": False,
}

detector = LogBERTDetector(
    model_path=options["model_path"],
    vocab_path=options["vocab_path"],
    device=options["device"]
)

# Predictor needs more options; extend existing dict (unused fields given safe defaults)
predictor_options = {
    "model_path": "./output/hdfs/bert/best_bert.pth",
    "vocab_path": "./output/hdfs/vocab.pkl",
    "device": "cpu",
    "window_size": 5,
    "adaptive_window": True,
    "seq_len": 20,
    "corpus_lines": 1,          # single-line prediction
    "on_memory": True,
    "batch_size": 32,           # unused in single-line path
    "num_workers": 0,
    "num_candidates": 6,
    "output_dir": "./output/hdfs/",
    "model_dir": "./output/hdfs/bert/",
    "gaussian_mean": 0.0,       # not used unless calling Predictor.predict()
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

predictor = Predictor(predictor_options)  # instantiate Predictor

class LogRequest(BaseModel):
    log_line: str

@app.post("/predict")
def predict(req: LogRequest):
    result = detector.predict(req.log_line, options)
    anomaly = any(r["anomaly"] for r in result)
    return {
        "input": req.log_line,
        "results": result,
        "is_anomaly": anomaly
    }

class LinePredictRequest(BaseModel):
    log_line: str
    seq_threshold: float = 0.9
    include_details: bool = False

@app.post("/predict_line")
def predict_line(req: LinePredictRequest):
    """
    Return simple anomaly decision for one raw log line.
    seq_threshold: ratio (undetected / masked) above which a window is anomalous.
    include_details: include per-window stats when True.
    """
    result = predictor.predict_single_log(req.log_line, seq_threshold=req.seq_threshold)
    return {
        "input": req.log_line,
        "is_anomaly": result["is_anomaly"],
        "anomalous_sequences": result["anomaly_sequences"],
        "total_sequences": result["total_sequences"],
        "seq_threshold": req.seq_threshold,
        "details": result["sequences"] if req.include_details else None
    }

# Optional: additional alias endpoint
class SingleLineRequest(BaseModel):
    log_line: str
    seq_threshold: float = 0.5

@app.post("/predict_single_line")
def predict_single_line(req: SingleLineRequest):
    r = predictor.predict_single_log(req.log_line, seq_threshold=req.seq_threshold)
    return {"is_anomaly": r["is_anomaly"], "anomalous_sequences": r["anomaly_sequences"], "total_sequences": r["total_sequences"]}
