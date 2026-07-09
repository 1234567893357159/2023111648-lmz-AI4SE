"""
步骤三 Task 2: CodeBERT Merge Prediction 在 AI 代码上的推理评估
只加载 lab3 微调好的模型，不重新训练
"""

import json
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

import importlib
import config
importlib.reload(config)

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class MergeDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": item["label"],
        }


class CodeBERTEvaluator:
    """CodeBERT 推理评估器：加载 lab3 微调好的模型 + 对 AI 代码做预测"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.tokenizer = None
        self.model = None
        self.results = {}
        self.y_true = None
        self.y_pred = None
        self.y_proba = None

    def load_model(self):
        print("加载 lab3 CodeBERT 模型...")
        model_dir = config.LAB3_CODEBERT_MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"  模型路径: {model_dir}")

    def _build_text(self, pr):
        parts = []
        title = pr.get("title", "")
        if title:
            parts.append(f"title: {title}")
        body = pr.get("body", "")
        if body:
            parts.append(f"body: {body}")
        commits = pr.get("commits", [])
        if commits:
            commit_msgs = [c.get("message", "") for c in commits]
            parts.append(f"commits: {' | '.join(commit_msgs)}")
        files = pr.get("files", [])
        if files:
            filenames = [f.get("filename", "") for f in files]
            parts.append(f"files: {', '.join(filenames)}")
        patches = []
        for f in files:
            patch = f.get("patch", "")
            if patch:
                patches.append(patch)
        text = "\n".join(parts)
        if patches:
            text += "\n" + "\n".join(patches)
        return text

    def prepare_data(self):
        print("准备 AI PR 数据...")
        ai_prs = []
        with open(config.AI_PRS_PATH, "r", encoding="utf-8") as f:
            import pandas as pd
            ai_df = pd.read_csv(f)
        ai_pr_ids = set(ai_df["pr_id"].tolist())

        samples = []
        raw_dir = config.LAB1_RAW_DIR
        for fname in sorted(os.listdir(raw_dir)):
            if not fname.endswith("_pulls.json"):
                continue
            with open(os.path.join(raw_dir, fname), "r", encoding="utf-8") as f:
                prs = json.load(f)
            for pr in prs:
                if pr["pr_id"] in ai_pr_ids:
                    text = self._build_text(pr)
                    label = 1 if pr.get("merged", False) else 0
                    samples.append({"text": text, "label": label, "pr_id": pr["pr_id"]})

        print(f"  AI PR 样本数: {len(samples)}")
        print(f"  Merged: {sum(s['label'] for s in samples)}")
        return samples

    def predict_and_evaluate(self, samples):
        print("推理预测...")
        dataset = MergeDataset(samples, self.tokenizer, config.CODEBERT_MAX_LENGTH)
        loader = DataLoader(dataset, batch_size=config.CODEBERT_BATCH_SIZE, shuffle=False)

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="CodeBERT 推理"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(batch["label"])

        logits = torch.cat(all_logits).numpy()
        self.y_true = np.array([l.item() for batch in all_labels for l in batch])
        probs = torch.sigmoid(torch.tensor(logits)).numpy().flatten()
        self.y_proba = probs
        self.y_pred = (probs > 0.5).astype(int)

        self.results = {
            "accuracy": float(accuracy_score(self.y_true, self.y_pred)),
            "precision": float(precision_score(self.y_true, self.y_pred, zero_division=0)),
            "recall": float(recall_score(self.y_true, self.y_pred, zero_division=0)),
            "f1": float(f1_score(self.y_true, self.y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(self.y_true, self.y_proba)),
        }

        cm = confusion_matrix(self.y_true, self.y_pred)
        print(f"\n  样本数: {len(self.y_true)}")
        print(f"  Accuracy:    {self.results['accuracy']:.4f}")
        print(f"  Precision:   {self.results['precision']:.4f}")
        print(f"  Recall:      {self.results['recall']:.4f}")
        print(f"  F1-score:    {self.results['f1']:.4f}")
        print(f"  ROC-AUC:     {self.results['roc_auc']:.4f}")
        print(f"  混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

    def save_results(self):
        output = {
            "n_samples": int(len(self.y_true)),
            "n_merged": int(self.y_true.sum()),
            "n_not_merged": int((1 - self.y_true).sum()),
            "model": "CodeBERT",
            "metrics": self.results,
        }
        lab3_res_path = os.path.join(config.LAB3_BASE_DIR, "data", "results", "classification_results.json")
        if os.path.exists(lab3_res_path):
            with open(lab3_res_path, "r", encoding="utf-8") as f:
                lab3_res = json.load(f)
            if "CodeBERT" in lab3_res:
                output["human_baseline"] = lab3_res["CodeBERT"]
            else:
                output["human_baseline"] = {}

        with open(config.STEP3_TASK2_CODEBERT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {config.STEP3_TASK2_CODEBERT_PATH}")

    def plot_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("CodeBERT Merge Prediction: AI Code Performance", fontsize=16, fontweight="bold")

        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Merged", "Merged"],
            yticklabels=["Not Merged", "Merged"],
            ax=axes[0],
        )
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title("Confusion Matrix - CodeBERT (AI Code)", fontsize=13, fontweight="bold")

        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)
        axes[1].plot(fpr, tpr, lw=2, color="#4dabf7", label=f"CodeBERT (AUC={auc:.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve - CodeBERT (AI Code)", fontsize=13, fontweight="bold")
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        fig_path = os.path.join(config.LAB5_FIGURES_DIR, "09_step3_codebert_comparison.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"图表已保存: {fig_path}")

    def run_all(self):
        print("=" * 60)
        print("步骤三 Task 2: CodeBERT Merge Prediction 推理评估")
        print("=" * 60)
        self.load_model()
        samples = self.prepare_data()
        self.predict_and_evaluate(samples)
        self.save_results()
        self.plot_comparison()
        print("\n" + "=" * 60)
        print("Task 2 完成!")
        print("=" * 60)
        return self.results

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CodeBERT 结果摘要")
        print("=" * 60)
        print(f"\n样本数: {len(self.y_true)}")
        print(f"  Merged: {self.y_true.sum()} ({self.y_true.sum()/len(self.y_true):.1%})")
        for k, v in self.results.items():
            print(f"  {k:>10}: {v:.4f}")