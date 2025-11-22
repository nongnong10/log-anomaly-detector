import os
from HDFS import data_process

def main():
    # Override globals used by mapping()
    data_process.output_dir = 'dataset/hdfs/test_parser/'
    data_process.log_file = 'my_log_data.log'

    # Run mapping()
    result_path = data_process.mapping()
    if result_path:
        print("Mapping JSON path:", result_path)
        try:
            with open(result_path, 'r') as f:
                preview = f.read(500)
            print("Preview (first 500 chars):")
            print(preview)
        except Exception as e:
            print("Failed to read mapping JSON:", e)
    else:
        print("mapping() returned None")

if __name__ == "__main__":
    main()

