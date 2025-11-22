import os
import re
import json
import argparse
import hashlib
import pandas as pd
from logparser import Drain
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- Configuration paths (adjust if layout differs) ----
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output', 'hdfs')
TEMPLATES_CSV = os.path.join(OUTPUT_DIR, 'HDFS.log_templates.csv')
MAPPING_JSON = os.path.join(OUTPUT_DIR, 'hdfs_log_templates.json')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'bert')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_bert.pth')
VOCAB_PATH = os.path.join(OUTPUT_DIR, 'vocab.pkl')
SEQUENCE_FILENAME = 'my_log_data_sequence'              # relative to OUTPUT_DIR
RESULT_FILE = os.path.join(OUTPUT_DIR, 'test_result')   # optional export file
LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

# Minimal regex list (same as test file)
REGEX = [
    r"(?<=blk_)[-\d]+",
    r"\d+\.\d+\.\d+\.\d+",
    r"(/[-\w]+)+",
]

def parse_raw_log(raw_log_path):
    """
    Step 1: Use Drain parse() to produce structured log and templates (warm start from TEMPLATES_CSV).
    Returns path to structured CSV.
    """
    if not os.path.isfile(raw_log_path):
        raise FileNotFoundError(f"Raw log not found: {raw_log_path}")
    if not os.path.isfile(TEMPLATES_CSV):
        print(f"Warm templates not found (will start cold): {TEMPLATES_CSV}")

    indir = os.path.dirname(raw_log_path)
    log_file = os.path.basename(raw_log_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = Drain.LogParser(
        log_format=LOG_FORMAT,
        indir=indir,
        outdir=OUTPUT_DIR,
        depth=5,
        st=0.5,
        rex=REGEX,
        keep_para=False
    )
    parser.parse(
        log_file,
        previous_templates_csv=TEMPLATES_CSV if os.path.isfile(TEMPLATES_CSV) else None,
        previous_structured_csv=None
    )
    structured_csv = os.path.join(OUTPUT_DIR, log_file + '_structured.csv')
    if not os.path.isfile(structured_csv):
        raise RuntimeError("Structured CSV not generated.")
    print(f"[STEP 1] Structured log: {structured_csv}")
    return structured_csv

def build_event_sequences(structured_csv):
    """
    Build block-level event sequences using mapping JSON.
    Output file: OUTPUT_DIR / SEQUENCE_FILENAME (each line: space-separated event integers).
    """
    if not os.path.isfile(MAPPING_JSON):
        raise FileNotFoundError(f"Mapping JSON not found: {MAPPING_JSON}")
    with open(MAPPING_JSON, 'r') as f:
        mapping = json.load(f)  # EventId -> int code

    rows = []
    with open(structured_csv, newline='') as f:
        rows = list(csv_dict_reader(f))

    blk_regex = re.compile(r'(blk_-?\d+)')
    block_map = {}
    for r in rows:
        eid = r.get('EventId')
        if not eid:
            continue
        mapped = mapping.get(eid, -1)
        if mapped == -1:
            continue
        blk_ids = set(blk_regex.findall(r.get('Content', '')))
        for b in blk_ids:
            block_map.setdefault(b, []).append(mapped)

    seq_path = os.path.join(OUTPUT_DIR, SEQUENCE_FILENAME)
    with open(seq_path, 'w') as f:
        for _, seq in block_map.items():
            if seq:
                f.write(' '.join(map(str, seq)) + '\n')
    print(f"[STEP 1] Sequence file: {seq_path}")
    return SEQUENCE_FILENAME  # return relative name for Predictor

def csv_dict_reader(fh):
    import csv
    return csv.DictReader(fh)

def init_predictor():
    """
    Prepare options dict required by Predictor.
    """
    device = 'cpu'
    options = {
        "model_path": MODEL_PATH,
        "vocab_path": VOCAB_PATH,
        "device": device,
        "window_size": 128,
        "adaptive_window": True,
        "seq_len": 512,
        "corpus_lines": None,
        "on_memory": True,
        "batch_size": 32,
        "num_workers": 2,
        "num_candidates": 6,
        "output_dir": OUTPUT_DIR + '/',   # Predictor concatenates
        "model_dir": MODEL_DIR + '/',
        "gaussian_mean": 0,
        "gaussian_std": 1,
        "is_logkey": True,
        "is_time": False,
        "scale_path": os.path.join(MODEL_DIR, 'scale.pkl'),
        "hypersphere_loss": False,
        "hypersphere_loss_test": False,
        "test_ratio": 1,
        "mask_ratio": 0.65,
        "min_len": 10
    }
    for p in [MODEL_PATH, VOCAB_PATH]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required model asset missing: {p}")
    try:
        from bert_pytorch.predict_log import Predictor  # local import to delay torch load
    except Exception as e:
        raise ImportError(
            "Failed to import Predictor / torch. If you still see 'libtorch_cpu.so: cannot enable executable stack', "
            "try: docker run --security-opt seccomp=unconfined ... or use the official PyTorch base image (already applied). "
            f"Original error: {e}"
        )
    return Predictor(options)

def run_pipeline(raw_log_path, seq_threshold=0.5, export=False):
    """
    Full pipeline: parse -> sequence -> predict -> summary.
    Returns summary dict.
    """
    structured_csv = parse_raw_log(raw_log_path)
    sequence_rel_name = build_event_sequences(structured_csv)
    predictor = init_predictor()
    result = predictor.predict_file(sequence_rel_name, seq_threshold=seq_threshold)
    is_anomaly = len(result["anomaly_indices"]) > 0
    print(f"[STEP 2] Anomaly detected: {is_anomaly}")
    summary = {
        "raw_log": raw_log_path,
        "sequence_file": os.path.join(OUTPUT_DIR, sequence_rel_name),
        "total_sequences": result["total_sequences"],
        "anomalous_sequences": len(result["anomaly_indices"]),
        "anomaly_ratio": result["anomaly_ratio"],
        "threshold_ratio": seq_threshold,
        "is_anomaly": is_anomaly,
        "anomaly_indices": result.get("anomaly_indices", [])
    }
    if export:
        with open(RESULT_FILE, 'w') as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")
        print(f"Result exported: {RESULT_FILE}")
    return summary

def detect_anomaly_from_raw(raw_log_data: str, seq_threshold: float = 0.2):
    """
    Endpoint helper: run anomaly detection directly from raw log text.
    Creates a temporary log file, then reuses run_pipeline.
    """
    if not raw_log_data.strip():
        raise ValueError("Empty raw_log_data provided.")
    with tempfile.TemporaryDirectory() as td:
        tmp_log = os.path.join(td, "input.log")
        with open(tmp_log, 'w') as f:
            f.write(raw_log_data)
        return run_pipeline(tmp_log, seq_threshold=seq_threshold, export=False)

def main():
    parser = argparse.ArgumentParser(description="End-to-end log anomaly pipeline")
    parser.add_argument("--log", default=os.path.join(PROJECT_ROOT, "dataset", "hdfs", "my_log_data.log"),
                        help="Path to raw HDFS log file")
    parser.add_argument("--seq_threshold", type=float, default=0.2,
                        help="Ratio threshold for sequence anomaly detection")
    parser.add_argument("--export", action="store_true", help="Export summary file")
    args = parser.parse_args()
    run_pipeline(args.log, seq_threshold=args.seq_threshold, export=args.export)

if __name__ == "__main__":
    main()
