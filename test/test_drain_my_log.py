import os
from logparser import Drain
import tempfile
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
REGEX = [
    r"(?<=blk_)[-\d]+",      # block id
    r'\d+\.\d+\.\d+\.\d+',   # IP
    r"(/[-\w]+)+",           # file path
]

ST = 0.5
DEPTH = 5

def resolve_paths():
    return {
        "INPUT_DIR": os.path.join(PROJECT_ROOT, 'dataset', 'hdfs'),
        "OUTPUT_DIR": os.path.join(PROJECT_ROOT, 'dataset', 'hdfs', 'test_parser'),
        "LOG_FILE": 'my_log_data.log',
        "PREV_TEMPLATES": os.path.join(PROJECT_ROOT, 'logbert', 'output', 'hdfs', 'HDFS.log_templates.csv'),
        "PREV_STRUCTURED": os.path.join(PROJECT_ROOT, 'logbert', 'output', 'hdfs', 'HDFS.log_structured.csv'),
    }

paths = resolve_paths()
INPUT_DIR = paths["INPUT_DIR"]
OUTPUT_DIR = paths["OUTPUT_DIR"]
LOG_FILE = paths["LOG_FILE"]
PREV_TEMPLATES = paths["PREV_TEMPLATES"]
PREV_STRUCTURED = paths["PREV_STRUCTURED"]
RAW_LOG_FILE = os.path.join(INPUT_DIR, LOG_FILE)  # absolute path to input log

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(RAW_LOG_FILE):
        raise FileNotFoundError(f"Input log file not found: {RAW_LOG_FILE}")
    parser = Drain.LogParser(
        log_format=LOG_FORMAT,
        indir=INPUT_DIR,
        outdir=OUTPUT_DIR,
        depth=DEPTH,
        st=ST,
        rex=REGEX,
        keep_para=False
    )

    # Optional: show how many old clusters will be loaded
    if os.path.isfile(PREV_TEMPLATES):
        old_clusters = parser.load_previous_clusters(PREV_TEMPLATES, PREV_STRUCTURED)
        print(f"Warm start clusters: {len(old_clusters)}")

    parser.parse(
        LOG_FILE,
        previous_templates_csv=PREV_TEMPLATES,
        previous_structured_csv=PREV_STRUCTURED
    )

    print("Done.")
    print(f"Structured: {os.path.join(OUTPUT_DIR, LOG_FILE + '_structured.csv')}")
    print(f"Templates:  {os.path.join(OUTPUT_DIR, LOG_FILE + '_templates.csv')}")

# --- Added test function below ---
def test_main_parses_single_line():
    """Lightweight test for main() end-to-end parsing with dynamic project paths."""
    with tempfile.TemporaryDirectory() as td:
        global INPUT_DIR, OUTPUT_DIR, LOG_FILE, PREV_TEMPLATES, PREV_STRUCTURED
        INPUT_DIR = os.path.join(td, 'dataset', 'hdfs')
        OUTPUT_DIR = os.path.join(td, 'dataset', 'hdfs', 'test_parser')
        os.makedirs(INPUT_DIR, exist_ok=True)
        LOG_FILE = 'sample.log'
        PREV_TEMPLATES = os.path.join(td, 'warm', 'HDFS.log_templates.csv')  # will not exist
        PREV_STRUCTURED = os.path.join(td, 'warm', 'HDFS.log_structured.csv')

        # Create a minimal log line matching LOG_FORMAT
        log_line = "081109 203615 143 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_123 terminating\n"
        with open(os.path.join(INPUT_DIR, LOG_FILE), 'w') as f:
            f.write(log_line)

        # Act
        main()

        # Assert: output files exist
        structured_path = os.path.join(OUTPUT_DIR, LOG_FILE + '_structured.csv')
        templates_path = os.path.join(OUTPUT_DIR, LOG_FILE + '_templates.csv')
        assert os.path.isfile(structured_path), "Structured CSV not created"
        assert os.path.isfile(templates_path), "Templates CSV not created"

        df_struct = pd.read_csv(structured_path)
        assert len(df_struct) == 1, "Expected exactly one parsed line"
        assert df_struct.loc[0, 'Content'].startswith('PacketResponder'), "Content column mismatch"
        assert df_struct.loc[0, 'EventTemplate'], "EventTemplate should not be empty"

        df_templ = pd.read_csv(templates_path)
        assert df_templ.shape[0] >= 1, "At least one template expected"

        print("test_main_parses_single_line: OK")

# --- Added real path test function ---
def test_main_real_file():
    """Test main() using actual project directories (creates sample log if missing)."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    if not os.path.isfile(RAW_LOG_FILE):
        sample = "081109 203615 143 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_999 terminating\n"
        with open(RAW_LOG_FILE, "w") as f:
            f.write(sample)
    main()
    structured_path = os.path.join(OUTPUT_DIR, LOG_FILE + "_structured.csv")
    templates_path = os.path.join(OUTPUT_DIR, LOG_FILE + "_templates.csv")
    assert os.path.isfile(structured_path), "Structured CSV missing"
    assert os.path.isfile(templates_path), "Templates CSV missing"
    df_struct = pd.read_csv(structured_path)
    assert len(df_struct) >= 1, "No parsed lines"
    assert "EventTemplate" in df_struct.columns, "EventTemplate column missing"
    print("test_main_real_file: OK")

if __name__ == "__main__":
    test_main_real_file()
    # test_main_parses_single_line()  # optional secondary test
