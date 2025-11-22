import os
import pandas as pd
from tqdm import tqdm
from HDFS import data_process

def main():
    # Override globals to point to prior Drain output
    data_process.output_dir = 'dataset/hdfs/test_parser/'
    data_process.log_file = 'my_log_data.log'
    data_process.input_dir = '../dataset/hdfs/'  # for anomaly_label.csv

    out_dir = data_process.output_dir
    log_file = data_process.log_file
    structured_csv = os.path.join(out_dir, log_file + '_structured.csv')
    templates_csv = os.path.join(out_dir, log_file + '_templates.csv')
    mapping_json = os.path.join(out_dir, 'hdfs_log_templates.json')
    sequence_csv = os.path.join(out_dir, 'hdfs_sequence.csv')

    # Check prerequisites
    if not os.path.isfile(structured_csv):
        print(f"Structured log missing: {structured_csv}")
        print("Run test_drain_my_log.py first.")
        return

    if not os.path.isfile(templates_csv):
        print(f"Templates CSV missing: {templates_csv}")
        print("Run test_drain_my_log.py first.")
        return

    # Create mapping if absent
    # if not os.path.isfile(mapping_json):
    #     print("Mapping JSON not found. Generating...")
    #     data_process.mapping()
    # else:
    #     print("Mapping JSON exists.")

    # Create sequence file if absent
    # if not os.path.isfile(sequence_csv):
    #     print("Sequence CSV not found. Generating via hdfs_sampling...")
    #     data_process.hdfs_sampling(structured_csv)
    # else:
    #     print("Sequence CSV exists.")

    # Run train/test generation
    print("Generating train/test splits...")
    # data_process.generate_train_test(sequence_csv, n=50)  # small sample for test
    n = 50
    ratio = 0.8
    blk_label_dict = {}
    blk_label_file = os.path.join(data_process.input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(sequence_csv)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    print("Printing labeled records:")
    for _, r in seq.iterrows():
        print(f"[Labeled] BlockId={r['BlockId']} Label={r['Label']} EventSequence={r['EventSequence']}")

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)
    print("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    # train = normal_seq.iloc[:train_len]
    # test_normal = normal_seq.iloc[train_len:]
    # test_abnormal = abnormal_seq
    #
    # df_to_file(train, data_process.output_dir + "train")
    # df_to_file(test_normal, data_process.output_dir + "test_normal")
    # df_to_file(test_abnormal, data_process.output_dir + "test_abnormal")
    # print("generate train test data done")
    #
    # # Output file summaries
    # for name in ["train", "test_normal", "test_abnormal"]:
    #     path = os.path.join(out_dir, name)
    #     print(f"{name} -> {path} {'(ok)' if os.path.isfile(path) else '(missing)'}")

def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')

if __name__ == "__main__":
    main()
