"""
数据集划分模块
从 lab2 数据集中收集有效 PR，按比例划分为训练集、验证集、测试集
每个样本包含 pr_id、before_path、after_path、label（label=1 表示已合并，0 表示未合并）
"""

import json
import os
import random
from typing import List, Dict, Tuple

import sys
sys.path.insert(0, "..")
import config


class DatasetSplitter:
    """数据集划分器"""

    def __init__(self):
        self.samples: List[Dict] = []
        self.stats = {
            "total": 0,
            "positive": 0,
            "negative": 0,
            "skipped_no_code": 0,
        }

    def _has_valid_files(self, dir_path: str) -> bool:
        """检查目录是否至少有一个有效代码文件（跳过 .notfound）"""
        if not os.path.exists(dir_path):
            return False
        files = os.listdir(dir_path)
        if not files:
            return False
        has_valid = any(not f.endswith(".notfound") for f in files)
        return has_valid

    def _load_pr_list(self, owner: str, repo: str) -> List[Dict]:
        """从 raw json 文件加载 PR 列表"""
        repo_key = f"{owner}_{repo}_pulls.json"
        json_path = os.path.join(config.LAB2_HUMAN_RAW_DIR, repo_key)
        if not os.path.exists(json_path):
            print(f"  ⚠️ 文件不存在: {json_path}，跳过")
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            pr_list = json.load(f)
        return pr_list

    def _collect_repo_samples(self, owner: str, repo: str) -> None:
        """收集单个仓库的所有有效样本"""
        repo_key = f"{owner}_{repo}"
        pr_list = self._load_pr_list(owner, repo)
        if not pr_list:
            return

        code_base_dir = os.path.join(config.LAB2_CODE_DIR, "human", repo_key)
        if not os.path.exists(code_base_dir):
            print(f"  ⚠️ 代码目录不存在: {code_base_dir}，跳过")
            return

        print(f"  处理 {len(pr_list)} 个 PR...")

        for pr in pr_list:
            pr_id = pr["pr_id"]
            merged = pr.get("merged", False)
            label = 1 if merged else 0

            pr_dir = os.path.join(code_base_dir, str(pr_id))
            before_dir = os.path.join(pr_dir, "before")
            after_dir = os.path.join(pr_dir, "after")

            if not self._has_valid_files(before_dir) or not self._has_valid_files(after_dir):
                self.stats["skipped_no_code"] += 1
                continue

            rel_before = os.path.join("data", "code", "human", repo_key, str(pr_id), "before")
            rel_after = os.path.join("data", "code", "human", repo_key, str(pr_id), "after")

            self.samples.append({
                "pr_id": pr_id,
                "repo": repo_key,
                "before_path": rel_before,
                "after_path": rel_after,
                "merged": merged,
                "label": label,
            })

            if merged:
                self.stats["positive"] += 1
            else:
                self.stats["negative"] += 1
            self.stats["total"] += 1

    def _split_dataset(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """按照 80%/10%/10% 比例划分数据集"""
        random.seed(config.RANDOM_SEED)
        shuffled = self.samples.copy()
        random.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * config.TRAIN_RATIO)
        n_val = int(n_total * config.VAL_RATIO)

        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

        return train, val, test

    def _save_split(self, data: List[Dict], filepath: str) -> None:
        """保存划分结果到 JSON 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _count_stats(self, data: List[Dict]) -> Tuple[int, int]:
        """统计数据集中正负样本数量"""
        pos = sum(1 for item in data if item["label"] == 1)
        neg = len(data) - pos
        return pos, neg

    def print_stats(self, train: List[Dict], val: List[Dict], test: List[Dict]) -> None:
        """打印统计信息"""
        train_pos, train_neg = self._count_stats(train)
        val_pos, val_neg = self._count_stats(val)
        test_pos, test_neg = self._count_stats(test)

        print("=" * 60)
        print("数据集划分完成")
        print("=" * 60)
        print(f"  总有效样本: {self.stats['total']}")
        print(f"    已合并(label=1): {self.stats['positive']}")
        print(f"    未合并(label=0): {self.stats['negative']}")
        print(f"    跳过(无代码): {self.stats['skipped_no_code']}")
        print("-" * 60)
        print(f"  训练集 ({config.TRAIN_RATIO * 100:.0f}%): {len(train)} 样本")
        print(f"    已合并: {train_pos}, 未合并: {train_neg}")
        print(f"  验证集 ({config.VAL_RATIO * 100:.0f}%): {len(val)} 样本")
        print(f"    已合并: {val_pos}, 未合并: {val_neg}")
        print(f"  测试集 ({(1 - config.TRAIN_RATIO - config.VAL_RATIO) * 100:.0f}%): {len(test)} 样本")
        print(f"    已合并: {test_pos}, 未合并: {test_neg}")
        print("-" * 60)
        print(f"  保存路径:")
        print(f"    {config.TRAIN_JSON_PATH}")
        print(f"    {config.VAL_JSON_PATH}")
        print(f"    {config.TEST_JSON_PATH}")
        print("=" * 60)

    def run(self) -> Dict[str, List[Dict]]:
        """运行完整划分流程"""
        print("=" * 60)
        print("开始数据集划分...")
        print("=" * 60)

        for owner, repo in config.TARGET_REPOS:
            print(f"\n处理仓库: {owner}/{repo}")
            self._collect_repo_samples(owner, repo)

        train, val, test = self._split_dataset()

        self._save_split(train, config.TRAIN_JSON_PATH)
        self._save_split(val, config.VAL_JSON_PATH)
        self._save_split(test, config.TEST_JSON_PATH)

        self.print_stats(train, val, test)

        return {
            "train": train,
            "val": val,
            "test": test,
        }


if __name__ == "__main__":
    splitter = DatasetSplitter()
    splitter.run()