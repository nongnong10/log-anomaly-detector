import os
import torch
from bert_pytorch.predict_log import Predictor

def ensure_dummy_log_file(output_dir, filename):
    path = os.path.join(output_dir, filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.isfile(path):
        # Minimal dummy log lines (space-separated token ids or placeholders)
        with open(path, "w") as f:
            f.write("1 2 3 4 5\n")
            f.write("2 3 4 6 7 8\n")
            f.write("10 11 12 13 14 15 16\n")
    return path

def build_options():
    return {
        "model_path": "/home/huy/logbert/output/hdfs/bert/best_bert.pth",          # adjust
        "vocab_path": "/home/huy/logbert/output/hdfs/vocab.pkl",          # adjust
        "device": "cpu",
        "window_size": 10,
        "adaptive_window": True,
        "seq_len": 100,
        "corpus_lines": None,
        "on_memory": True,
        "batch_size": 16,
        "num_workers": 0,
        "num_candidates": 3,
        "output_dir": "/home/huy/logbert/output/hdfs/",                   # must contain my_log_data
        "model_dir": "/home/huy/logbert/output/hdfs/bert/",                  # for center / error_dict
        "gaussian_mean": 0.0,
        "gaussian_std": 1.0,
        "is_logkey": True,
        "is_time": False,
        "scale_path": "/home/huy/logbert/models/scale.pkl",        # only used if is_time=True
        "hypersphere_loss": False,
        "hypersphere_loss_test": False,
        "test_ratio": 1,       # use all sequences
        "mask_ratio": 0.3,
        "min_len": 1
    }

def main():
    options = build_options()
    # ensure_dummy_log_file(options["output_dir"], "my_log_data")  # uncomment if file may not exist
    predictor = Predictor(options)
    predictor.predict_file("my_log_data", seq_threshold=0.1)

if __name__ == "__main__":
    main()
