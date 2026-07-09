"""
步骤4a：CodeBERT 数据准备
从 lab1 raw JSON 提取 PR 的 title、body、commits、files.patch，
按 train/val/test 划分保存到 lab3 本地
"""

import json
import os

import sys
sys.path.insert(0, "..")
import config


class CodeBERTDataPreparer:
    """从 lab1 原始数据提取 CodeBERT 所需字段"""

    def __init__(self):
        self._pr_cache = {}

    def _load_lab1_prs(self, repo_key: str) -> dict:
        """加载 lab1 某个仓库的 PR 数据，建立 pr_id → PR 的索引"""
        if repo_key in self._pr_cache:
            return self._pr_cache[repo_key]

        json_path = os.path.join(config.LAB1_RAW_DIR, f"{repo_key}_pulls.json")
        if not os.path.exists(json_path):
            print(f"  ⚠️ 文件不存在: {json_path}")
            self._pr_cache[repo_key] = {}
            return {}

        with open(json_path, "r", encoding="utf-8") as f:
            pr_list = json.load(f)

        index = {pr["pr_id"]: pr for pr in pr_list}
        self._pr_cache[repo_key] = index
        return index

    def _build_text(self, pr: dict) -> str:
        """将 PR 的 title、body、commits、files.patch 拼接成一条文本"""
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

    def prepare_split(self, split_path: str, output_path: str, desc: str) -> None:
        """处理一个划分（train/val/test），提取字段并保存"""
        print(f"\n处理 {desc} 数据...")

        with open(split_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        results = []
        missing = 0

        for sample in samples:
            pr_id = sample["pr_id"]
            repo = sample["repo"]
            label = sample["label"]

            pr_index = self._load_lab1_prs(repo)
            pr = pr_index.get(pr_id)

            if pr is None:
                missing += 1
                continue

            text = self._build_text(pr)
            results.append({
                "pr_id": pr_id,
                "repo": repo,
                "label": label,
                "text": text,
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        pos = sum(1 for r in results if r["label"] == 1)
        neg = len(results) - pos
        print(f"  {desc}: {len(results)} 样本 (正:{pos}, 负:{neg}), 缺失: {missing}")

    def run(self) -> None:
        """运行完整数据准备流程"""
        print("=" * 60)
        print("CodeBERT 数据准备：从 lab1 提取 PR 信息")
        print("=" * 60)

        self.prepare_split(config.TRAIN_JSON_PATH, config.CODEBERT_TRAIN_PATH, "训练集")
        self.prepare_split(config.VAL_JSON_PATH, config.CODEBERT_VAL_PATH, "验证集")
        self.prepare_split(config.TEST_JSON_PATH, config.CODEBERT_TEST_PATH, "测试集")

        print("\n" + "=" * 60)
        print("数据准备完成")
        print(f"  训练集: {config.CODEBERT_TRAIN_PATH}")
        print(f"  验证集: {config.CODEBERT_VAL_PATH}")
        print(f"  测试集: {config.CODEBERT_TEST_PATH}")
        print("=" * 60)


if __name__ == "__main__":
    preparer = CodeBERTDataPreparer()
    preparer.run()