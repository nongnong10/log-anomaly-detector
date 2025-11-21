import sys
sys.path.append("../")
sys.path.append("../../")

import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../deeplog')


import argparse
import torch
import csv
import json
import re
from bert_pytorch.dataset import WordVocab
from bert_pytorch import Predictor, Trainer
from bert_pytorch.dataset.utils import seed_everything
from bert_pytorch.predict_log import Predictor
from HDFS import data_process

options = dict()
options['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
options["output_dir"] = "../output/hdfs/"
options["model_dir"] = options["output_dir"] + "bert/"
options["model_path"] = options["model_dir"] + "best_bert.pth"
options["train_vocab"] = options["output_dir"] + "train"
options["vocab_path"] = options["output_dir"] + "vocab.pkl"  # pickle file

options["window_size"] = 128
options["adaptive_window"] = True
options["seq_len"] = 512
options["max_len"] = 512 # for position embedding
options["min_len"] = 10
options["mask_ratio"] = 0.65
# sample ratio
options["train_ratio"] = 1
options["valid_ratio"] = 0.1
options["test_ratio"] = 1

# features
options["is_logkey"] = True
options["is_time"] = False

options["hypersphere_loss"] = True
options["hypersphere_loss_test"] = False

options["scale"] = None # MinMaxScaler()
options["scale_path"] = options["model_dir"] + "scale.pkl"

# model
options["hidden"] = 256 # embedding size
options["layers"] = 4
options["attn_heads"] = 4

options["epochs"] = 200
options["n_epochs_stop"] = 10
options["batch_size"] = 32

options["corpus_lines"] = None
options["on_memory"] = True
options["num_workers"] = 5
options["lr"] = 1e-3
options["adam_beta1"] = 0.9
options["adam_beta2"] = 0.999
options["adam_weight_decay"] = 0.00
options["with_cuda"]= True
options["cuda_devices"] = None
options["log_freq"] = None

# predict
options["num_candidates"] = 6
options["gaussian_mean"] = 0
options["gaussian_std"] = 1

seed_everything(seed=1234)

if not os.path.exists(options['model_dir']):
    os.makedirs(options['model_dir'], exist_ok=True)

print("device", options["device"])
print("features logkey:{} time: {}\n".format(options["is_logkey"], options["is_time"]))
print("mask ratio", options["mask_ratio"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(mode='train')

    predict_parser = subparsers.add_parser('predict')
    predict_parser.set_defaults(mode='predict')
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std", type=float, default=1)
    predict_parser.add_argument("--input_log", type=str, default=None,
                                help="Raw HDFS log file to predict (e.g. dataset/hdfs/my_log_data.log)")

    vocab_parser = subparsers.add_parser('vocab')
    vocab_parser.set_defaults(mode='vocab')
    vocab_parser.add_argument("-s", "--vocab_size", type=int, default=None)
    vocab_parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    vocab_parser.add_argument("-m", "--min_freq", type=int, default=1)

    args = parser.parse_args()
    print("arguments", args)

    if args.mode == 'train':
        Trainer(options).train()

    elif args.mode == 'predict':
        if args.input_log:
            raw_path = args.input_log
            if not os.path.isfile(raw_path):
                print(f"input_log not found: {raw_path}")
                sys.exit(1)
            # Build sequence file name
            seq_name = "predict_tmp"
            mapping_json = options["output_dir"] + "hdfs_log_templates.json"
            if not os.path.isfile(mapping_json):
                print("Template mapping not found. Run data_process.py first.")
                sys.exit(1)
            # Parse raw log with Drain into structured CSV
            log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
            custom_dir = os.path.dirname(raw_path)
            custom_file = os.path.basename(raw_path)
            data_process.parser(custom_dir, options["output_dir"], custom_file, log_format, 'drain')
            structured_csv = options["output_dir"] + custom_file + "_structured.csv"
            if not os.path.isfile(structured_csv):
                print("Structured CSV not generated.")
                sys.exit(1)
            # Load mapping
            with open(mapping_json, "r") as f:
                mapping_dict = json.load(f)
            # Build block sequences (same logic as hdfs_sampling but isolated)
            df = []
            with open(structured_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    df.append(row)
            block_map = {}
            for row in df:
                eid = mapping_dict.get(row["EventId"], -1)
                if eid == -1:
                    continue
                blk_ids = set(re.findall(r'(blk_-?\d+)', row["Content"]))
                for b in blk_ids:
                    block_map.setdefault(b, []).append(eid)
            seq_path = options["output_dir"] + seq_name
            with open(seq_path, "w") as f:
                for _, seq in block_map.items():
                    if seq:
                        f.write(" ".join(map(str, seq)) + "\n")
            print(f"Sequence file generated: {seq_path}")
            # Run prediction on sequence file
            Predictor(options).predict_file(seq_name)
        else:
            Predictor(options).predict()

    elif args.mode == 'vocab':
        with open(options["train_vocab"], "r", encoding=args.encoding) as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print("VOCAB SIZE:", len(vocab))
        print("save vocab in", options["vocab_path"])
        vocab.save_vocab(options["vocab_path"])
