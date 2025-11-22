import os
import re
import json
import threading
import time
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from logparser import Drain
from HDFS import data_process

LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
REGEX_PATTERNS = [
    r'(?<=blk_)[-\d]+',
    r'\d+\.\d+\.\d+\.\d+',
    r'(/[-\w]+)+',
]
ST = 0.5
DEPTH = 5

INPUT_DIR = '../dataset/hdfs/'
LABEL_FILE = os.path.join(INPUT_DIR, 'anomaly_label.csv')

app = FastAPI(title="HDFS Anomaly Endpoint Optimized")

# Global caches
_label_lock = threading.Lock()
_label_cache: Dict[str, int] = {}
_blk_regex = re.compile(r'(blk_-?\d+)')
_compiled_rex = REGEX_PATTERNS  # Drain expects pattern strings; keep list

# Single reusable working directory
_WORK_DIR = os.path.abspath('.cache_hdfs_parse')
os.makedirs(_WORK_DIR, exist_ok=True)

def load_labels_once() -> Dict[str, int]:
    with _label_lock:
        if _label_cache:
            return _label_cache
        if not os.path.isfile(LABEL_FILE):
            return {}
        df = pd.read_csv(LABEL_FILE, usecols=['BlockId', 'Label'])
        _label_cache.update({r.BlockId: 1 if r.Label == 'Anomaly' else 0 for r in df.itertuples()})
        return _label_cache

# Lazy parser factory (create new per request; can be optimized to reuse if library is stateless)
def build_parser(indir: str, outdir: str) -> Drain.LogParser:
    return Drain.LogParser(
        log_format=LOG_FORMAT,
        indir=indir,
        outdir=outdir,
        depth=DEPTH,
        st=ST,
        rex=_compiled_rex,
        keep_para=False
    )

class LogRequest(BaseModel):
    logs: List[str]

@app.post("/detect")
def detect(req: LogRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Fast exit for empty input
    if not req.logs:
        return {
            "blocks": [],
            "overall_anomaly": False,
            "total_blocks": 0,
            "execution_time_parse_sec": 0.0,
            "execution_time_total_sec": round(time.perf_counter() - t0, 6)
        }

    # Write input once
    in_file = os.path.join(_WORK_DIR, 'input.log')
    with open(in_file, 'w') as f:
        f.write('\n'.join(line.rstrip('\n') for line in req.logs))
        f.write('\n')

    parse_dir = os.path.join(_WORK_DIR, 'parsed')
    os.makedirs(parse_dir, exist_ok=True)

    # Parse
    tp_start = time.perf_counter()
    parser = build_parser(_WORK_DIR, parse_dir)
    parser.parse('input.log')
    parse_time = time.perf_counter() - tp_start

    structured_path = os.path.join(parse_dir, 'input.log_structured.csv')
    if not os.path.isfile(structured_path):
        return {"error": "Parsing failed"}

    # Read only required columns
    tr_start = time.perf_counter()
    df = pd.read_csv(
        structured_path,
        usecols=['Content', 'EventId'],
        dtype={'Content': 'string', 'EventId': 'string'},
        engine='c'
    )

    # Vectorized block extraction
    # Extract all blk ids: creates MultiIndex -> (row, match)
    extracted = df['Content'].str.extractall(_blk_regex)
    if extracted.empty:
        total_time = time.perf_counter() - t0
        return {
            "blocks": [],
            "overall_anomaly": False,
            "total_blocks": 0,
            "execution_time_parse_sec": round(parse_time, 6),
            "execution_time_total_sec": round(total_time, 6)
        }

    # Map row index to EventId then aggregate
    event_map = df['EventId']
    extracted['EventId'] = event_map.loc[extracted.index.get_level_values(0)].values
    blk_events_series = extracted.groupby(0)['EventId'].agg(list)

    # Labels (cached)
    label_dict = load_labels_once()

    blocks_info = []
    overall_anomaly = False
    for blk_id, events in blk_events_series.items():
        lbl = label_dict.get(blk_id, 0)
        if lbl == 1:
            overall_anomaly = True
        blocks_info.append({
            "blockId": blk_id,
            "events": events,
            "label": lbl,
            "anomaly": lbl == 1
        })

    transform_time = time.perf_counter() - tr_start
    total_time = time.perf_counter() - t0

    return {
        "blocks": blocks_info,
        "overall_anomaly": overall_anomaly,
        "total_blocks": len(blocks_info),
        "execution_time_parse_sec": round(parse_time, 6),
        "execution_time_transform_sec": round(transform_time, 6),
        "execution_time_total_sec": round(total_time, 6)
    }

@app.get("/health")
def health():
    return {"status": "ok"}