import torch
from bert_pytorch.dataset import WordVocab, LogDataset
from bert_pytorch.dataset.sample import fixed_window


class LogBERTDetector:
    def __init__(self, model_path, vocab_path, device="cpu"):
        self.device = device
        self.model = torch.load(model_path, map_location=torch.device(self.device))
        self.model.to(self.device).eval()
        self.vocab = WordVocab.load_vocab(vocab_path)

    def preprocess_raw_line(self, raw_line):
        # TODO: replace with real template mapping
        tokens = raw_line.strip().split()
        return tokens

    def build_time_seqs(self, log_seqs, use_time):
        if not use_time:
            return [[0] * len(seq) for seq in log_seqs]
        return [[0] * len(seq) for seq in log_seqs]

    def predict(self, raw_line, opts, seq_threshold=0.5):
        tokens = self.preprocess_raw_line(raw_line)
        log_seqs, _ = fixed_window(
            " ".join(tokens),
            opts["window_size"],
            adaptive_window=opts["adaptive_window"],
            seq_len=opts["seq_len"],
            min_len=opts["min_len"]
        )
        if not log_seqs:
            log_seqs = [tokens[:opts["seq_len"]]]

        tim_seqs = self.build_time_seqs(log_seqs, opts["is_time"])

        dataset = LogDataset(
            log_seqs, tim_seqs, self.vocab,
            seq_len=opts["seq_len"],
            corpus_lines=len(log_seqs),
            on_memory=True,
            predict_mode=True,
            mask_ratio=opts["mask_ratio"]
        )

        batch = dataset.collate_fn([dataset[i] for i in range(len(dataset))])
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            out = self.model(batch["bert_input"], batch["time_input"])
            mask_lm_output = out["logkey_output"]

        results = []
        for i in range(len(batch["bert_label"])):
            input_ids = batch["bert_input"][i]
            label_ids = batch["bert_label"][i]
            mask_pos = (label_ids > 0).nonzero(as_tuple=True)[0]
            masked_tokens = len(mask_pos)
            undetected = 0
            for pos in mask_pos:
                true_id = int(label_ids[pos].item())
                logits = mask_lm_output[i, pos]
                topk = torch.argsort(-logits)[:opts["num_candidates"]]
                topk_ids = [int(x.item()) for x in topk]
                if true_id not in topk_ids:
                    undetected += 1
            ratio = undetected / masked_tokens if masked_tokens else 0.0
            anomaly = (masked_tokens > 0 and undetected > masked_tokens * seq_threshold)
            results.append({
                "window_index": i,
                "masked_tokens": masked_tokens,
                "undetected_tokens": undetected,
                "ratio": ratio,
                "anomaly": anomaly
            })
        return results
