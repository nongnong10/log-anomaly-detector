import os, sys, re, json, csv
sys.path.append("../")
from bert_pytorch.predict_log import Predictor
from HDFS import data_process
from HDFS.logbert import options

def ensure_mapping():
    """Ensure hdfs_log_templates.json exists, create if missing."""
    out_dir = os.path.abspath(options["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    mapping_json = os.path.join(out_dir, "hdfs_log_templates.json")

    if os.path.isfile(mapping_json):
        print(f"Mapping JSON already exists: {mapping_json}")
        return True

    print("Mapping JSON missing. Generating from training log...")
    try:
        # Update data_process paths to use absolute output_dir
        orig_output = data_process.output_dir
        data_process.output_dir = out_dir
        data_process.log_structured_file = os.path.join(out_dir, data_process.log_file + "_structured.csv")
        data_process.log_templates_file = os.path.join(out_dir, data_process.log_file + "_templates.csv")
        data_process.log_sequence_file = os.path.join(out_dir, "hdfs_sequence.csv")

        # Parse if templates CSV doesn't exist
        if not os.path.isfile(data_process.log_templates_file):
            LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
            print(f"Parsing training log to generate templates...")
            data_process.parser(data_process.input_dir, out_dir, data_process.log_file, LOG_FORMAT, 'drain')

        if not os.path.isfile(data_process.log_templates_file):
            print(f"Failed to create templates CSV at {data_process.log_templates_file}")
            data_process.output_dir = orig_output
            return False

        # Generate mapping
        created_path = data_process.mapping()
        data_process.output_dir = orig_output

        if created_path and os.path.isfile(created_path):
            print(f"Mapping generation succeeded: {created_path}")
            return True
        else:
            print("Mapping generation failed (no path returned or file not created).")
            return False

    except Exception as e:
        print(f"ensure_mapping() exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_sequence(raw_log_path, seq_filename):
    if not os.path.isfile(raw_log_path):
        print(f"File not found: {raw_log_path}")
        return None
    if not ensure_mapping():
        print("Cannot proceed without mapping JSON.")
        return None
    out_dir = os.path.abspath(options["output_dir"])
    mapping_json = os.path.join(out_dir, "hdfs_log_templates.json")
    with open(mapping_json, "r") as f:
        mapping_dict = json.load(f)

    input_dir = os.path.dirname(raw_log_path)
    log_file = os.path.basename(raw_log_path)
    data_process.parser(input_dir, options["output_dir"], log_file, LOG_FORMAT, 'drain')
    structured_csv = options["output_dir"] + log_file + "_structured.csv"
    if not os.path.isfile(structured_csv):
        print("Structured CSV not produced.")
        return None

    block_map = {}
    with open(structured_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = mapping_dict.get(row["EventId"], -1)
            if eid == -1:
                continue
            blk_ids = set(re.findall(r'(blk_-?\d+)', row["Content"]))
            for b in blk_ids:
                block_map.setdefault(b, []).append(eid)

    seq_path = os.path.join(out_dir, seq_filename)
    with open(seq_path, "w") as f:
        for seq in block_map.values():
            if seq:
                f.write(" ".join(map(str, seq)) + "\n")
    print(f"Sequence file written: {seq_path}")
    return seq_filename

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_mylog.py dataset/hdfs/my_log_data.log")
        return
    raw_path = sys.argv[1]
    seq_name = build_sequence(raw_path, "predict_tmp")
    if seq_name:
        Predictor(options).predict_file(seq_name)

if __name__ == "__main__":
    main()
