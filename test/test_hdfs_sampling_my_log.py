import json
import os
from HDFS import data_process
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import re


def main():
    data_process.output_dir = 'dataset/hdfs/test_parser/'
    data_process.log_file = 'my_log_data.log'
    # structured log produced by test_drain_my_log.py
    structured_csv = os.path.join(data_process.output_dir, data_process.log_file + '_structured.csv')
    mapping_json = os.path.join(data_process.output_dir, 'hdfs_log_templates.json')
    sequence_csv = os.path.join(data_process.output_dir, 'hdfs_sequence.csv')

    if not os.path.isfile(structured_csv):
        print(f"Structured CSV missing: {structured_csv}")
        print("Run test_drain_my_log.py first.")
        return

    if not os.path.isfile(mapping_json):
        print("Mapping JSON missing. Generating...")
        result = data_process.mapping()
        if not result:
            print("Failed to create mapping; aborting.")
            return
    else:
        print("Mapping JSON exists.")

    print("Running hdfs_sampling...")

    window = 'session'
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", structured_csv)
    df = pd.read_csv(structured_csv, engine='c',
                     na_filter=False, memory_map=True, dtype={'Date': object, "Time": object})

    with open(data_process.output_dir + "hdfs_log_templates.json", "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict = defaultdict(list)  # preserve insertion order of items
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict[blk_Id].append(row["EventId"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    data_df.to_csv(sequence_csv, index=None)
    print("hdfs sampling done")

    if os.path.isfile(sequence_csv):
        print(f"Sequence file created: {sequence_csv}")
        try:
            df = pd.read_csv(sequence_csv)
            print("Preview:")
            print(df.head().to_string(index=False))
        except Exception as e:
            print("Failed to read sequence CSV preview:", e)
    else:
        print("Sequence file not found after sampling.")


if __name__ == "__main__":
    main()
