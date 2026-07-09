"""
词汇表构建模块
从训练集代码中提取 AST 路径，构建 token 和 path 的词汇表映射
"""

import json
import os
from collections import Counter

from tqdm import tqdm

import sys
sys.path.insert(0, "..")
import config
from ast_path_extractor import ASTPathExtractor


class VocabBuilder:
    """词汇表构建器，构建 token → id 和 path → id 的映射"""

    def __init__(self):
        self.extractor = ASTPathExtractor()
        self.token_counter = Counter()
        self.path_counter = Counter()
        self.token_to_id = {}
        self.path_to_id = {}

    def _get_cache_path(self, pr_id):
        return os.path.join(config.AST_CACHE_DIR, f"{pr_id}.json")

    def _try_load_cached_paths(self, cache_path):
        """尝试加载缓存中的路径，仅当 max_paths_cached 匹配当前配置时有效"""
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("max_paths_cached") == config.MAX_PATHS_PER_FILE:
                bp = data.get("before_paths", [])
                ap = data.get("after_paths", [])
                if bp and ap:
                    return [tuple(p) for p in bp], [tuple(p) for p in ap]
        except Exception:
            pass
        return None

    def _save_cache(self, cache_path, before_paths, after_paths):
        """保存 AST 路径到缓存"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = {
            "before_paths": [list(p) for p in before_paths],
            "after_paths": [list(p) for p in after_paths],
            "max_paths_cached": config.MAX_PATHS_PER_FILE,
            "version": 4,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _collect_paths_from_pr(self, pr_item: dict):
        """收集单个 PR 的 AST 路径（优先使用缓存，首次解析后缓存路径）"""
        pr_id = pr_item["pr_id"]
        cache_path = self._get_cache_path(pr_id)

        cached = self._try_load_cached_paths(cache_path)
        if cached is not None:
            before_paths, after_paths = cached
        else:
            base_dir = config.LAB2_BASE_DIR
            before_dir = os.path.join(base_dir, pr_item["before_path"])
            after_dir = os.path.join(base_dir, pr_item["after_path"])

            before_paths = self.extractor.extract_paths_from_dir(before_dir)
            after_paths = self.extractor.extract_paths_from_dir(after_dir)

            self._save_cache(cache_path, before_paths, after_paths)

        for start_token, path_str, end_token in (before_paths + after_paths):
            self.token_counter[start_token] += 1
            self.token_counter[end_token] += 1
            self.path_counter[path_str] += 1

    def build(self, train_prs: list):
        """从训练集 PR 构建词汇表"""
        print("构建词汇表：从训练集代码中提取 AST 路径...")
        for pr_item in tqdm(train_prs, desc="收集 AST 路径"):
            self._collect_paths_from_pr(pr_item)

        token_to_id = {"<PAD>": 0, "<UNK>": 1}
        for token, freq in self.token_counter.most_common():
            token_to_id[token] = len(token_to_id)

        path_to_id = {"<PAD>": 0}
        for path_str, freq in self.path_counter.most_common():
            if freq >= config.MIN_PATH_FREQ:
                path_to_id[path_str] = len(path_to_id)

        self.token_to_id = token_to_id
        self.path_to_id = path_to_id

        print(f"  Token 词汇表大小: {len(token_to_id)}")
        print(f"  路径词汇表大小: {len(path_to_id)} (过滤后，最小频率={config.MIN_PATH_FREQ})")

    def save(self):
        """保存词汇表到文件"""
        vocab = {
            "token_to_id": self.token_to_id,
            "path_to_id": self.path_to_id,
            "num_tokens": len(self.token_to_id),
            "num_paths": len(self.path_to_id),
        }
        os.makedirs(os.path.dirname(config.VOCAB_PATH), exist_ok=True)
        with open(config.VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        print(f"词汇表已保存到: {config.VOCAB_PATH}")

    def load(self):
        """从文件加载词汇表"""
        with open(config.VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        self.token_to_id = vocab["token_to_id"]
        self.path_to_id = vocab["path_to_id"]
        print(f"词汇表已加载: {len(self.token_to_id)} tokens, {len(self.path_to_id)} paths")

    def token_to_idx(self, token: str) -> int:
        """将 token 转换为 ID"""
        return self.token_to_id.get(token, 1)

    def path_to_idx(self, path_str: str) -> int:
        """将路径字符串转换为 ID，未知路径映射到 PAD"""
        return self.path_to_id.get(path_str, 0)