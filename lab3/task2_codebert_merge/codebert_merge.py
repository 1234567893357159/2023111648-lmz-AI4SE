"""
步骤4b：CodeBERT Merge Prediction 微调
使用 transformers Trainer API 微调 CodeBERT 进行二分类（是否合并）
"""

import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
sys.path.insert(0, "..")
import config


class MergeDataset(Dataset):
    """CodeBERT 合并预测数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }


def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy().flatten()
    labels = labels.astype(int).flatten()

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class CodeBERTMergeTrainer:
    """CodeBERT 合并预测微调器"""

    def __init__(self):
        self.model_name = config.CODEBERT_MODEL_NAME
        self.max_length = config.CODEBERT_MAX_LENGTH
        self.batch_size = config.CODEBERT_BATCH_SIZE
        self.learning_rate = config.CODEBERT_LEARNING_RATE
        self.epochs = config.CODEBERT_EPOCHS
        self.weight_decay = config.CODEBERT_WEIGHT_DECAY
        self.output_dir = config.CODEBERT_MODEL_DIR

        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _load_data(self):
        """加载数据集"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.train_dataset = MergeDataset(
            config.CODEBERT_TRAIN_PATH, self.tokenizer, self.max_length
        )
        self.val_dataset = MergeDataset(
            config.CODEBERT_VAL_PATH, self.tokenizer, self.max_length
        )
        self.test_dataset = MergeDataset(
            config.CODEBERT_TEST_PATH, self.tokenizer, self.max_length
        )

        print(f"  训练集: {len(self.train_dataset)} 样本")
        print(f"  验证集: {len(self.val_dataset)} 样本")
        print(f"  测试集: {len(self.test_dataset)} 样本")

    def train(self):
        """微调 CodeBERT"""
        print("=" * 60)
        print("CodeBERT 微调：Merge Prediction")
        print("=" * 60)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  使用设备: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 60)

        self._load_data()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            disable_tqdm=False,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("\n开始训练...")
        self.trainer.train()

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"\n模型已保存到: {self.output_dir}")

    def evaluate(self) -> dict:
        """在测试集上评估"""
        print("\n" + "=" * 60)
        print("测试集评估")
        print("=" * 60)

        if self.trainer is not None:
            predictions = self.trainer.predict(self.test_dataset)
            logits = predictions.predictions
            labels = predictions.label_ids
        else:
            self.model.eval()
            device = next(self.model.parameters()).device
            all_logits = []
            all_labels = []
            from torch.utils.data import DataLoader
            loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
            with torch.no_grad():
                for batch in loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels_batch = batch["labels"]
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    all_logits.append(outputs.logits.cpu())
                    all_labels.append(labels_batch)
            logits = torch.cat(all_logits).numpy()
            labels = torch.cat(all_labels).numpy()

        probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
        preds = (probs > 0.5).astype(int)

        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": preds,
            "probabilities": probs,
            "labels": labels,
        }

    def predict(self, texts: list) -> np.ndarray:
        """对文本列表进行预测"""
        self.model.eval()
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.sigmoid(outputs.logits).numpy().flatten()

        return probs

    def load(self):
        """加载已保存的模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.output_dir
        )
        print(f"模型已加载: {self.output_dir}")

        self.test_dataset = MergeDataset(
            config.CODEBERT_TEST_PATH, self.tokenizer, self.max_length
        )
        print(f"  测试集: {len(self.test_dataset)} 样本")


if __name__ == "__main__":
    trainer = CodeBERTMergeTrainer()
    trainer.train()
    trainer.evaluate()