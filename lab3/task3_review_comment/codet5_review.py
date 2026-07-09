"""
CodeT5 Review Comment Generation 微调
使用 CodeT5（Encoder-Decoder）进行代码审查意见自动生成
优化：1024 输入长度 + 8-bit Adam + 梯度累积
"""

import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config


class ReviewDataset(Dataset):
    """代码审查意见生成数据集"""

    def __init__(self, data_path: str, tokenizer, max_input_length: int = 512, max_target_length: int = 128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        target_text = item["target"]

        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


def compute_metrics(eval_pred, tokenizer):
    """计算 BLEU-4 和 ROUGE-L 指标"""
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    smooth = SmoothingFunction().method1
    bleu_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        if not pred or not label:
            bleu_scores.append(0.0)
        else:
            bleu = sentence_bleu(
                [label.split()],
                pred.split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
            bleu_scores.append(bleu)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        if not pred or not label:
            rouge_scores.append(0.0)
        else:
            scores = scorer.score(label, pred)
            rouge_scores.append(scores["rougeL"].fmeasure)

    return {
        "bleu": np.mean(bleu_scores),
        "rougeL": np.mean(rouge_scores),
    }


class CodeT5ReviewTrainer:
    """CodeT5 代码审查意见生成微调器"""

    def __init__(self):
        self.model_name = config.CODET5_MODEL_NAME
        self.max_input_length = config.CODET5_MAX_INPUT_LENGTH
        self.max_target_length = config.CODET5_MAX_TARGET_LENGTH
        self.batch_size = config.CODET5_BATCH_SIZE
        self.learning_rate = config.CODET5_LEARNING_RATE
        self.epochs = config.CODET5_EPOCHS
        self.weight_decay = config.CODET5_WEIGHT_DECAY
        self.warmup_steps = config.CODET5_WARMUP_STEPS
        self.gradient_accumulation_steps = config.CODET5_GRADIENT_ACCUMULATION_STEPS
        self.output_dir = config.CODET5_MODEL_DIR

        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _load_data(self):
        """加载数据集"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = self.max_input_length

        self.train_dataset = ReviewDataset(
            config.CODEREVIEW_TRAIN_PATH, self.tokenizer, self.max_input_length, self.max_target_length
        )
        self.val_dataset = ReviewDataset(
            config.CODEREVIEW_VAL_PATH, self.tokenizer, self.max_input_length, self.max_target_length
        )
        self.test_dataset = ReviewDataset(
            config.CODEREVIEW_TEST_PATH, self.tokenizer, self.max_input_length, self.max_target_length
        )

        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        print(f"  测试集: {len(self.test_dataset)} 样本")

    def train(self):
        """微调 CodeT5"""
        print("=" * 70)
        print("CodeT5 微调：Review Comment Generation")
        print("=" * 70)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  使用设备: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  输入长度: {self.max_input_length}")
        print(f"  目标长度: {self.max_target_length}")
        print("=" * 70)

        self._load_data()

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.gradient_checkpointing_enable()

        torch.cuda.empty_cache()

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            generation_num_beams=4,
            eval_accumulation_steps=4,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            disable_tqdm=False,
            fp16=torch.cuda.is_available(),
            optim="adamw_8bit",
            dataloader_num_workers=0,
        )

        def wrapped_compute_metrics(eval_pred):
            return compute_metrics(eval_pred, self.tokenizer)

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=wrapped_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("\n开始训练...")

        checkpoint = None
        if os.path.exists(self.output_dir):
            all_checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")]
            valid_checkpoints = []
            for d in all_checkpoints:
                ckpt_path = os.path.join(self.output_dir, d)
                if os.path.exists(os.path.join(ckpt_path, "model.safetensors")) or \
                   os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
                    valid_checkpoints.append(d)
            if valid_checkpoints:
                checkpoint = os.path.join(
                    self.output_dir,
                    sorted(valid_checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                )
                print(f"\n发现有效检查点: {checkpoint}，将从此恢复训练...")
            else:
                if all_checkpoints:
                    print(f"\n发现 {len(all_checkpoints)} 个检查点但均无效（缺少模型文件），将从头开始训练...")
                else:
                    print("\n未发现检查点，从头开始训练...")
        else:
            print("\n未发现检查点，从头开始训练...")

        self.trainer.train(resume_from_checkpoint=checkpoint)

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"\n模型已保存到: {self.output_dir}")

    def evaluate(self) -> dict:
        """在测试集上评估，返回预测结果"""
        print("\n" + "=" * 70)
        print("测试集评估 (Review Comment Generation)")
        print("=" * 70)

        if self.trainer is not None:
            predictions = self.trainer.predict(self.test_dataset)
            pred_ids = predictions.predictions
            label_ids = predictions.label_ids
        else:
            self.model.eval()
            device = next(self.model.parameters()).device
            all_pred_ids = []
            all_label_ids = []
            from torch.utils.data import DataLoader
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_batch = batch["labels"]
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=self.max_target_length,
                        num_beams=4,
                    )
                    all_pred_ids.append(generated.cpu())
                    all_label_ids.append(labels_batch)
            pred_ids = torch.cat(all_pred_ids).numpy()
            label_ids = torch.cat(all_label_ids).numpy()

        pred_ids = np.where(pred_ids != -100, pred_ids, self.tokenizer.pad_token_id)
        label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        smooth = SmoothingFunction().method1
        bleu_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            if not pred or not label:
                bleu_scores.append(0.0)
            else:
                bleu = sentence_bleu(
                    [label.split()],
                    pred.split(),
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth,
                )
                bleu_scores.append(bleu)

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            if not pred or not label:
                rouge_scores.append(0.0)
            else:
                scores = scorer.score(label, pred)
                rouge_scores.append(scores["rougeL"].fmeasure)

        avg_bleu = np.mean(bleu_scores)
        avg_rouge = np.mean(rouge_scores)

        print(f"  BLEU-4:   {avg_bleu:.4f}")
        print(f"  ROUGE-L:  {avg_rouge:.4f}")

        return {
            "bleu": avg_bleu,
            "rougeL": avg_rouge,
            "predictions": decoded_preds,
            "labels": decoded_labels,
        }

    def generate(self, texts: list) -> list:
        """对输入文本列表生成审查意见"""
        self.model.eval()
        device = next(self.model.parameters()).device
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def load(self):
        """加载已保存的模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        self.tokenizer.model_max_length = self.max_input_length
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.output_dir)
        print(f"模型已加载: {self.output_dir}")

        self.test_dataset = ReviewDataset(
            config.CODEREVIEW_TEST_PATH, self.tokenizer, self.max_input_length, self.max_target_length
        )
        print(f"  测试集: {len(self.test_dataset)} 样本")


if __name__ == "__main__":
    trainer = CodeT5ReviewTrainer()
    trainer.train()
    trainer.evaluate()