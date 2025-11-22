from bert_pytorch.predict_log import Predictor
from HDFS.predict_mylog import build_sequence, ensure_mapping
from HDFS.logbert import options
import os

RAW_LOG = "dataset/hdfs/my_log_data.log"
SEQ_NAME = "predict_tmp"

def run_prediction(raw_log_path=RAW_LOG):
    if not os.path.isfile(raw_log_path):
        print(f"Raw log not found: {raw_log_path}")
        return

    abs_out = os.path.abspath(options["output_dir"])
    print(f"Using output_dir: {abs_out}")

    # Ensure mapping exists
    if not ensure_mapping():
        print("ERROR: Mapping generation failed; aborting prediction.")
        print(f"Check that {abs_out} is writable and training data exists.")
        return

    # Build sequence file
    seq_file = build_sequence(raw_log_path, SEQ_NAME)
    if not seq_file:
        print("ERROR: Sequence build failed.")
        return

    # Run prediction
    print("\n--- Running prediction ---")
    result = Predictor(options).predict_file(seq_file)

    # Print summary
    print(f"\n=== PREDICTION SUMMARY ===")
    print(f"Total sequences: {result['total_sequences']}")
    print(f"Anomalous sequences: {result['anomalous_sequences']}")
    print(f"Anomaly ratio: {result['ratio']:.4f}")
    print(f"Threshold ratio used: {result['threshold_ratio']}")

    # Print per-sequence details
    print(f"\n=== SEQUENCE DETAILS ===")
    anomalous_indices = []
    for i, r in enumerate(result["details"]):
        masked = max(r["masked_tokens"], 1)
        ratio = r["undetected_tokens"] / masked
        is_anom = (r["undetected_tokens"] > r["masked_tokens"] * 0.5) or (r.get("deepSVDD_label", 0) == 1)
        if is_anom:
            anomalous_indices.append(i)
        print(f"Seq {i}: masked={r['masked_tokens']} undetected={r['undetected_tokens']} "
              f"ratio={ratio:.2f} deepSVDD={r.get('deepSVDD_label', 0)} anomaly={is_anom}")

    print(f"\nAnomalous sequence indices: {anomalous_indices}")

if __name__ == "__main__":
    run_prediction()
