import os
import re
import json
import argparse
import hashlib
import pandas as pd

from database.get_log_lines import count_anomaly_log_lines, get_log_lines
from logparser import Drain
import tempfile
from .detect_anomaly_sequence import init_predictor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.upsert_log_block import upsert_log_block, batch_upsert_log_blocks
from database.upsert_anomaly_sequence import batch_upsert_anomaly_sequences
from .notifiy_slack import send_slack_alert

# Global variables to store block mappings for anomaly score updates
BLOCK_EVENT_IDS = {}
BLOCK_ID_LIST = []

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def parse_raw_log_v2(raw_log_path, db_conn=None):
    """
    Parse raw log to structured CSV using Drain algorithm.
    If db_conn provided, also store log lines in database by block_id.
    Returns the structured CSV file path.
    """
    global BLOCK_ID_LIST
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
    if db_conn:
        print("=== parse_raw_log_v2 - flow parse_and_store_log_lines")
        BLOCK_ID_LIST = parser.parse_and_store_log_lines(log_file, warm_start=True, db_conn=db_conn)
    else:
        print("=== parse_raw_log_v2 - flow parse (old flow)")
    parser.parse(
            log_file,
            previous_templates_csv=TEMPLATES_CSV if os.path.isfile(TEMPLATES_CSV) else None,
            previous_structured_csv=None
        )
    structured_csv = os.path.join(OUTPUT_DIR, log_file + '_structured.csv')
    if not os.path.isfile(structured_csv):
        raise RuntimeError("Structured CSV not generated.")
    return structured_csv

def build_event_sequences_v2(structured_csv, db_conn=None):
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
    block_event_ids = {}
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
            block_event_ids.setdefault(b, []).append(eid)

    seq_path = os.path.join(OUTPUT_DIR, SEQUENCE_FILENAME)

    # Prepare globals and batch data
    batch_data = []

    with open(seq_path, 'w') as f:
        for block_id, seq in block_map.items():
            if seq:
                f.write(' '.join(map(str, seq)) + '\n')

                # Collect globals for later mapping and for prediction mapping
                # BLOCK_ID_LIST.append(block_id)
                # Store original EventId sequence for DB storage and later score updates
                event_sequence = block_event_ids.get(block_id, [])
                BLOCK_EVENT_IDS[block_id] = event_sequence

                # If DB provided, accumulate batch entry (initial anomaly_score = 0.0)
                if db_conn:
                    has_data = len(event_sequence) > 0
                    anomaly_score = 0.0
                    batch_data.append((block_id, event_sequence, has_data, anomaly_score))

    # If DB connection provided, perform a single batch upsert for all blocks
    if db_conn and batch_data:
        success = batch_upsert_log_blocks(db_conn, batch_data)
        if success:
            print(f"Batch upserted {len(batch_data)} blocks to database")
        else:
            print("Batch upsert failed for initial log_block insertions")

    # print("block_map", block_map)
    # print("block_event_ids", block_event_ids)
    # print(f"[STEP 1] Sequence file: {seq_path}")
    return SEQUENCE_FILENAME  # return relative name for Predictor


def build_event_sequences_v3(structured_csv, db_conn=None):
    """
    Build block-level event sequences using mapping JSON.
    Output file: OUTPUT_DIR / SEQUENCE_FILENAME (each line: space-separated event integers).
    """
    if not os.path.isfile(MAPPING_JSON):
        raise FileNotFoundError(f"Mapping JSON not found: {MAPPING_JSON}")
    with open(MAPPING_JSON, 'r') as f:
        mapping = json.load(f)  # EventId -> int code

    block_map = {}
    block_event_ids = {}
    block_ids = BLOCK_ID_LIST
    for block_id in block_ids:
        log_lines = get_log_lines(db_conn, block_id)
        for log_line in log_lines:
            mapped = mapping.get(log_line.event_id, -1)
            if mapped == -1:
                continue
            block_map.setdefault(block_id, []).append(mapped)
            block_event_ids.setdefault(block_id, []).append(log_line.event_id)

    seq_path = os.path.join(OUTPUT_DIR, SEQUENCE_FILENAME)
    batch_data = []

    with open(seq_path, 'w') as f:
        for block_id, seq in block_map.items():
            if seq:
                f.write(' '.join(map(str, seq)) + '\n')
                event_sequence = block_event_ids.get(block_id, [])
                if db_conn:
                    has_data = len(event_sequence) > 0
                    anomaly_score = 0.0
                    batch_data.append((block_id, event_sequence, has_data, anomaly_score))

    # If DB connection provided, perform a single batch upsert for all blocks
    if db_conn and batch_data:
        print("batch_upsert_log_blocks sequence for block: ", batch_data)
        success = batch_upsert_log_blocks(db_conn, batch_data)
        if success:
            print(f"Batch upserted {len(batch_data)} blocks to database")
        else:
            print("Batch upsert failed for initial log_block insertions")
    return SEQUENCE_FILENAME  # return relative name for Predictor


def csv_dict_reader(fh):
    import csv
    return csv.DictReader(fh)

def run_pipeline_v2(raw_log_path, seq_threshold=0.5, export=False, db_conn=None, notify_slack: bool = False):
    """
    Full pipeline: parse -> sequence -> predict -> summary.
    Returns summary dict.
    notify_slack: if True, send Slack alerts for anomalous sequences
    """
    # Step 1: Parse raw log to structured CSV (now also stores log lines in DB)
    print("==== [STEP 1] Parsing raw log...")
    structured_csv = parse_raw_log_v2(raw_log_path, db_conn=db_conn)
    print(f"==== [STEP 1] Structured log: {structured_csv} \n")

    # Step 2: Build event sequences (initial DB upsert for log_block)
    print("==== [STEP 2] Building event sequences...")
    # sequence_rel_name = build_event_sequences_v2(structured_csv, db_conn=db_conn)
    sequence_rel_name = build_event_sequences_v3(structured_csv, db_conn=db_conn)
    print(f"==== [STEP 2] Event sequences file: {sequence_rel_name} \n")

    # Step 3: Predict anomalies
    predictor = init_predictor()
    print("==== [STEP 3] Starting prediction...")
    result = predictor.predict_file(sequence_rel_name, seq_threshold=seq_threshold)
    print("BLOCK_ID_LIST:", BLOCK_ID_LIST)
    print(f"==== [STEP 3] Prediction result: {result} \n")

    # Step 4: Batch update anomaly scores in database
    anomaly_block_score = {}
    print("==== [STEP 4] Updating anomaly scores in database...")
    if db_conn and BLOCK_ID_LIST:
        batch_data = []
        batch_anomaly_sequence_data = []
        results = result.get("results", [])
        for i, block_id in enumerate(BLOCK_ID_LIST):
            if i < len(results):
                res = results[i]
                undetected_tokens = res.get("undetected_tokens", 0)
                masked_tokens = res.get("masked_tokens", 1)  # Avoid division by zero
                anomaly_score = undetected_tokens / masked_tokens
            else:
                anomaly_score = 0.0  # Default score if no result is available

            count_anomaly_log_line = count_anomaly_log_lines(db_conn, block_id)
            # print(f"Block id {block_id}: {count_anomaly_log_line}")
            if count_anomaly_log_line.total_anomalous_lines > 0:
                anomaly_score = count_anomaly_log_line.total_anomalous_lines / count_anomaly_log_line.total_line
            # print(f"Block id {block_id} anomaly_score: {anomaly_score}")
            anomaly_block_score[block_id] = anomaly_score
            batch_data.append((block_id, None, None, anomaly_score))

            label = "Anomaly" if anomaly_score >= seq_threshold else "Normal"
            batch_anomaly_sequence_data.append((block_id, label))

        if batch_data and batch_anomaly_sequence_data:
            print("Batch data for log_block:", batch_data)
            success = batch_upsert_log_blocks(db_conn, batch_data) and batch_upsert_anomaly_sequences(db_conn, batch_anomaly_sequence_data)
            if success:
                print(f"Batch updated anomaly_score for {len(batch_data)} blocks")
            else:
                print("Batch update failed for anomaly scores")
    else:
        if not db_conn:
            print("No db_conn provided; skipping DB update for anomaly scores")
        else:
            print("Empty BLOCK_ID_LIST; nothing to update in DB")

    # Step 5: Summarize results in the requested response format
    print("==== [STEP 5] Summarizing results...")
    results = result.get("results", [])
    anomalous_sequences = []
    normal_sequences = []
    for i, block_id in enumerate(BLOCK_ID_LIST):
        anomaly_score = anomaly_block_score.get(block_id, 0.0)
        if i < len(results):
            res = results[i]
            undetected_tokens = res.get("undetected_tokens", 0)
            masked_tokens = res.get("masked_tokens", 1)
        is_anomaly = anomaly_score >= seq_threshold
        seq_obj = {
            "block_id": block_id,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly
        }
        if is_anomaly:
            anomalous_sequences.append(seq_obj)
            # Only notify Slack if explicitly requested
            if notify_slack:
                try:
                    send_slack_alert(block_id, anomaly_score, is_anomaly)
                except Exception as e:
                    print(f"[run_pipeline_v2] send_slack_alert failed for {block_id}: {e}")
        else:
            normal_sequences.append(seq_obj)

    summary = {
        "anomaly_ratio": len(anomalous_sequences) / len(BLOCK_ID_LIST),
        "threshold_ratio": seq_threshold,
        "total_sequences": len(BLOCK_ID_LIST),
        "total_anomalous_sequences": len(anomalous_sequences)
    }
    response = {
        "anomalous_sequences": anomalous_sequences,
        "normal_sequences": normal_sequences,
        "summary": summary
    }

    if export:
        with open(RESULT_FILE, 'w') as f:
            json.dump(response, f, indent=2)
        print(f"Result exported: {RESULT_FILE}")

    return response

def detect_anomaly_from_raw_v2(raw_log_data: str, seq_threshold: float = 0.2, db_conn=None, notify_slack: bool = False):
    """
    Endpoint helper: run anomaly detection directly from raw log text.
    Creates a temporary log file, then reuses run_pipeline_v2.
    notify_slack: if True, send Slack alerts for anomalous sequences
    """
    if not raw_log_data.strip():
        raise ValueError("Empty raw_log_data provided.")
    with tempfile.TemporaryDirectory() as td:
        tmp_log = os.path.join(td, "input.log")
        with open(tmp_log, 'w') as f:
            f.write(raw_log_data)
        return run_pipeline_v2(tmp_log, seq_threshold=seq_threshold, export=False, db_conn=db_conn, notify_slack=notify_slack)

def main():
    parser = argparse.ArgumentParser(description="End-to-end log anomaly pipeline")
    parser.add_argument("--log", default=os.path.join(PROJECT_ROOT, "dataset", "hdfs", "my_log_data.log"),
                        help="Path to raw HDFS log file")
    parser.add_argument("--seq_threshold", type=float, default=0.2,
                        help="Ratio threshold for sequence anomaly detection")
    parser.add_argument("--export", action="store_true", help="Export summary file")
    args = parser.parse_args()
    run_pipeline_v2(args.log, seq_threshold=args.seq_threshold, export=args.export)

if __name__ == "__main__":
    main()
