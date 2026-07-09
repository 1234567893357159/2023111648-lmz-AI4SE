"""
代码向量化主流程
将步骤一划分的数据集中每个 PR 的 before/after 代码通过 Code2Vec 转化为向量
"""

import json
import os

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, "..")
import config
from ast_path_extractor import ASTPathExtractor
from vocab_builder import VocabBuilder
from code2vec_model import create_model


class CodeVectorizer:
    """代码向量化器，将 PR 代码转化为 Code2Vec 向量"""

    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.extractor = ASTPathExtractor()
        self.vocab = None
        self.model = None
        print(f"使用设备: {self.device}")

    def _load_pr_list(self, json_path: str):
        """加载 PR 列表"""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _paths_to_ids(self, paths: list, max_paths: int):
        """将 AST 路径列表转换为 ID 张量"""
        if not paths:
            return torch.zeros(3, 1, dtype=torch.long)

        n = min(len(paths), max_paths)
        starts = torch.zeros(n, dtype=torch.long)
        path_ids = torch.zeros(n, dtype=torch.long)
        ends = torch.zeros(n, dtype=torch.long)

        for i in range(n):
            start_token, path_str, end_token = paths[i]
            starts[i] = self.vocab.token_to_idx(start_token)
            path_ids[i] = self.vocab.path_to_idx(path_str)
            ends[i] = self.vocab.token_to_idx(end_token)

        return torch.stack([starts, path_ids, ends], dim=0)

    def _get_ast_cache_path(self, pr_id: int):
        """获取单个 PR 的 AST 缓存文件路径"""
        return os.path.join(config.AST_CACHE_DIR, f"{pr_id}.json")

    def _load_cache(self, cache_path: str):
        """加载缓存，返回 before_paths 和 after_paths

        仅在 max_paths_cached 匹配当前配置时才返回路径，
        否则返回 (None, None) 触发重新解析。
        """
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("max_paths_cached") == config.MAX_PATHS_PER_FILE:
            bp = data.get("before_paths", [])
            ap = data.get("after_paths", [])
            if bp and ap:
                return [tuple(p) for p in bp], [tuple(p) for p in ap]

        return None, None

    def _save_cache(self, cache_path: str, before_paths: list, after_paths: list):
        """保存 AST 路径到缓存"""
        data = {
            "before_paths": [list(p) for p in before_paths],
            "after_paths": [list(p) for p in after_paths],
            "max_paths_cached": config.MAX_PATHS_PER_FILE,
            "version": 4,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def _vectorize_one_pr(self, pr_item: dict):
        """向量化单个 PR

        缓存策略：
        - 若缓存存在且路径匹配 → 直接读路径（最快）
        - 若缓存不存在或不匹配 → 解析代码，缓存路径
        """
        pr_id = pr_item["pr_id"]
        cache_path = self._get_ast_cache_path(pr_id)

        before_paths, after_paths = None, None

        if os.path.exists(cache_path):
            before_paths, after_paths = self._load_cache(cache_path)

        if before_paths is None:
            base_dir = config.LAB2_BASE_DIR
            before_dir = os.path.join(base_dir, pr_item["before_path"])
            after_dir = os.path.join(base_dir, pr_item["after_path"])

            before_paths = self.extractor.extract_paths_from_dir(before_dir)
            after_paths = self.extractor.extract_paths_from_dir(after_dir)

            self._save_cache(cache_path, before_paths, after_paths)

        before_ids = self._paths_to_ids(before_paths, config.MAX_PATHS_PER_FILE)
        after_ids = self._paths_to_ids(after_paths, config.MAX_PATHS_PER_FILE)

        return before_ids, after_ids

    def _vectorize_dataset(self, pr_list: list, desc: str):
        """向量化一整个数据集"""
        pr_ids = []
        labels = []
        before_vectors = []
        after_vectors = []

        self.model.eval()
        with torch.no_grad():
            for pr_item in tqdm(pr_list, desc=desc):
                before_ids, after_ids = self._vectorize_one_pr(pr_item)

                before_ids_batch = before_ids.unsqueeze(0).to(self.device)
                after_ids_batch = after_ids.unsqueeze(0).to(self.device)

                before_vec = self.model(
                    before_ids_batch[:, 0, :],
                    before_ids_batch[:, 1, :],
                    before_ids_batch[:, 2, :],
                ).cpu()
                after_vec = self.model(
                    after_ids_batch[:, 0, :],
                    after_ids_batch[:, 1, :],
                    after_ids_batch[:, 2, :],
                ).cpu()

                pr_ids.append(pr_item["pr_id"])
                labels.append(pr_item["label"])
                before_vectors.append(before_vec)
                after_vectors.append(after_vec)

        pr_ids = torch.tensor(pr_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        before_vectors = torch.cat(before_vectors, dim=0)
        after_vectors = torch.cat(after_vectors, dim=0)

        return {
            "pr_ids": pr_ids,
            "labels": labels,
            "before_vectors": before_vectors,
            "after_vectors": after_vectors,
        }

    def _save_vectors(self, vectors_dict: dict, filepath: str):
        """保存向量到 .pt 文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(vectors_dict, filepath)
        n = len(vectors_dict["pr_ids"])
        print(f"  已保存 {n} 个样本到: {filepath}")

    def _all_vectors_exist(self) -> bool:
        """检查所有向量文件是否已存在"""
        return (
            os.path.exists(config.TRAIN_VECTORS_PATH)
            and os.path.exists(config.VAL_VECTORS_PATH)
            and os.path.exists(config.TEST_VECTORS_PATH)
        )

    def _vectorize_or_skip(self, pr_list: list, filepath: str, desc: str, force: bool):
        """向量化数据集，若文件已存在则跳过"""
        if not force and os.path.exists(filepath):
            data = torch.load(filepath, weights_only=False)
            print(f"  {desc}已存在 ({len(data['pr_ids'])} 个样本)，跳过")
            return data

        data = self._vectorize_dataset(pr_list, desc)
        self._save_vectors(data, filepath)
        return data

    def run(self, force: bool = False):
        """运行完整向量化流程

        Args:
            force: 是否强制重新生成（忽略已存在的文件）
        """
        print("=" * 60)
        print("开始 Code2Vec 代码向量化...")
        print("=" * 60)

        if not force and self._all_vectors_exist():
            print("\n所有向量文件已存在，跳过全部向量化:")
            print(f"  {config.TRAIN_VECTORS_PATH}")
            print(f"  {config.VAL_VECTORS_PATH}")
            print(f"  {config.TEST_VECTORS_PATH}")
            print("\n如需重新生成，请设置 force=True 或手动删除上述文件。")
            print("=" * 60)
            return

        print("\n[1/4] 加载数据集...")
        train_prs = self._load_pr_list(config.TRAIN_JSON_PATH)
        val_prs = self._load_pr_list(config.VAL_JSON_PATH)
        test_prs = self._load_pr_list(config.TEST_JSON_PATH)
        print(f"  训练集: {len(train_prs)} 个 PR")
        print(f"  验证集: {len(val_prs)} 个 PR")
        print(f"  测试集: {len(test_prs)} 个 PR")

        print("\n[2/4] 构建词汇表（仅使用训练集）...")
        if not force and os.path.exists(config.VOCAB_PATH):
            print(f"  词汇表已存在，跳过: {config.VOCAB_PATH}")
            self.vocab = VocabBuilder()
            self.vocab.load()
        else:
            self.vocab = VocabBuilder()
            self.vocab.build(train_prs)
            self.vocab.save()

        print("\n[3/4] 创建 Code2Vec 模型...")
        self.model = create_model(self.vocab)
        self.model = self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  模型参数量: {total_params:,}")

        print("\n[4/4] 向量化数据集...")
        train_vectors = self._vectorize_or_skip(train_prs, config.TRAIN_VECTORS_PATH, "训练集向量", force)
        val_vectors = self._vectorize_or_skip(val_prs, config.VAL_VECTORS_PATH, "验证集向量", force)
        test_vectors = self._vectorize_or_skip(test_prs, config.TEST_VECTORS_PATH, "测试集向量", force)

        print("\n" + "=" * 60)
        print("向量化完成！")
        print("=" * 60)
        shape_info = []
        if train_vectors is not None:
            shape_info.append(f"训练集: {train_vectors['before_vectors'].shape}")
        if val_vectors is not None:
            shape_info.append(f"验证集: {val_vectors['before_vectors'].shape}")
        if test_vectors is not None:
            shape_info.append(f"测试集: {test_vectors['before_vectors'].shape}")
        for s in shape_info:
            print(f"  {s}")
        print(f"  向量维度: {config.CODE_VECTOR_DIM}")
        print("=" * 60)


if __name__ == "__main__":
    vectorizer = CodeVectorizer()
    vectorizer.run()