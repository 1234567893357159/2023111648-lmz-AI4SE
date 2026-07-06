"""
步骤三：特征提取
从 PR 信息、AST、CFG 提取特征，聚合 + 均值序列化，before+after 并列
输出到 data/features/human_features.csv
"""

import os
import json
import csv
from tqdm import tqdm
from collections import defaultdict

import config


class FeatureExtractor:
    def __init__(self):
        self.stats = {
            "total_pr": 0,
            "success": 0,
            "failed": 0,
            "skipped_no_ast": 0,
        }
        self.output_csv = os.path.join(config.FEATURES_DIR, "human_features.csv")
        self.ast_types = []
        self.cfg_types = []

    def _collect_node_types(self):
        """收集所有出现过的 AST 和 CFG 节点类型，固定特征维度"""
        ast_types = set()
        cfg_types = set()

        root = config.HUMAN_AST_DIR
        for root_dir, _, files in os.walk(root):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(root_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for node in data.get("control_flow_nodes", []):
                        ast_types.add(node["type"])
                except Exception:
                    continue

        root = config.HUMAN_CFG_DIR
        for root_dir, _, files in os.walk(root):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(root_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for func in data.get("functions", []):
                        for node in func["graph"].get("nodes", []):
                            cfg_types.add(node["type"])
                except Exception:
                    continue

        self.ast_types = sorted(ast_types)
        self.cfg_types = sorted(cfg_types)
        print(f"收集到 AST 节点类型: {len(self.ast_types)} 种")
        print(f"收集到 CFG 节点类型: {len(self.cfg_types)} 种")

    def _get_pr_base_path(self, repo_key, pr_id):
        """获取 PR 各种目录的基础路径"""
        return {
            "code_before": os.path.join(config.HUMAN_CODE_DIR, repo_key, str(pr_id), "before"),
            "code_after": os.path.join(config.HUMAN_CODE_DIR, repo_key, str(pr_id), "after"),
            "ast_before": os.path.join(config.HUMAN_AST_DIR, repo_key, str(pr_id), "before"),
            "ast_after": os.path.join(config.HUMAN_AST_DIR, repo_key, str(pr_id), "after"),
            "cfg_before": os.path.join(config.HUMAN_CFG_DIR, repo_key, str(pr_id), "before"),
            "cfg_after": os.path.join(config.HUMAN_CFG_DIR, repo_key, str(pr_id), "after"),
        }

    def _extract_base_features(self, pr_data):
        """提取基础特征"""
        title = pr_data.get("title", "") or ""
        body = pr_data.get("body", "") or ""
        return {
            "file_count": pr_data.get("changed_files", 0),
            "additions": pr_data.get("additions", 0),
            "deletions": pr_data.get("deletions", 0),
            "title_len": len(title),
            "body_len": len(body),
        }

    def _extract_ast_from_dir(self, dir_path):
        """从一个目录（before/after）提取 AST 特征"""
        counts = {t: 0 for t in self.ast_types}
        total_nodes = 0
        file_count = 0

        if not os.path.exists(dir_path):
            return counts, total_nodes, file_count

        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for node in data.get("control_flow_nodes", []):
                    if node["type"] in counts:
                        counts[node["type"]] += 1
                total_nodes += data.get("node_count", 0)
                file_count += 1
            except Exception:
                continue

        return counts, total_nodes, file_count

    def _extract_ast_features(self, repo_key, pr_id):
        """提取 before 和 after AST 特征"""
        paths = self._get_pr_base_path(repo_key, pr_id)

        results = {}
        for version, dir_path in [("before", paths["ast_before"]), ("after", paths["ast_after"])]:
            counts, total_nodes, file_count = self._extract_ast_from_dir(dir_path)
            divide_by = max(file_count, 1)
            for t in self.ast_types:
                results[f"ast_{version}_{t}_sum"] = counts[t]
                results[f"ast_{version}_{t}_mean"] = counts[t] / divide_by
            results[f"ast_{version}_node_count_sum"] = total_nodes
            results[f"ast_{version}_node_count_mean"] = total_nodes / divide_by

        return results

    def _extract_cfg_from_dir(self, dir_path):
        """从一个目录（before/after）提取 CFG 特征"""
        type_counts = {t: 0 for t in self.cfg_types}
        total_nodes = 0
        total_edges = 0
        function_count = 0

        if not os.path.exists(dir_path):
            return type_counts, total_nodes, total_edges, function_count

        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for func in data.get("functions", []):
                    for node in func["graph"].get("nodes", []):
                        if node["type"] in type_counts:
                            type_counts[node["type"]] += 1
                    nodes = func["graph"].get("nodes", [])
                    edges = func["graph"].get("edges", [])
                    total_nodes += len(nodes)
                    total_edges += len(edges)
                    function_count += 1
            except Exception:
                continue

        return type_counts, total_nodes, total_edges, function_count

    def _extract_cfg_features(self, repo_key, pr_id):
        """提取 before 和 after CFG 特征"""
        paths = self._get_pr_base_path(repo_key, pr_id)

        results = {}
        for version, dir_path in [("before", paths["cfg_before"]), ("after", paths["cfg_after"])]:
            type_counts, total_nodes, total_edges, func_count = self._extract_cfg_from_dir(dir_path)
            divide_by = max(func_count, 1)
            for t in self.cfg_types:
                results[f"cfg_{version}_{t}_sum"] = type_counts[t]
                results[f"cfg_{version}_{t}_mean"] = type_counts[t] / divide_by

            results[f"cfg_{version}_total_nodes_sum"] = total_nodes
            results[f"cfg_{version}_total_nodes_mean"] = total_nodes / divide_by
            results[f"cfg_{version}_total_edges_sum"] = total_edges
            results[f"cfg_{version}_total_edges_mean"] = total_edges / divide_by
            results[f"cfg_{version}_function_count"] = func_count
            if func_count > 0:
                results[f"cfg_{version}_avg_nodes_per_func"] = total_nodes / func_count
                results[f"cfg_{version}_avg_edges_per_func"] = total_edges / func_count
                cyclomatic = total_edges - total_nodes + 2 * func_count
                results[f"cfg_{version}_cyclomatic_complexity"] = cyclomatic / func_count
            else:
                results[f"cfg_{version}_avg_nodes_per_func"] = 0
                results[f"cfg_{version}_avg_edges_per_func"] = 0
                results[f"cfg_{version}_cyclomatic_complexity"] = 0

        return results

    def _build_feature_row(self, pr_id, repo, base_feats, ast_feats, cfg_feats):
        """按固定顺序拼接特征行"""
        row = {
            "pr_id": pr_id,
            "repo": repo,
        }
        row.update(base_feats)
        row.update(ast_feats)
        row.update(cfg_feats)
        return row

    def _get_column_order(self):
        """生成固定列顺序"""
        cols = ["pr_id", "repo", "file_count", "additions", "deletions", "title_len", "body_len"]
        for version in ["before", "after"]:
            for t in self.ast_types:
                cols.append(f"ast_{version}_{t}_sum")
                cols.append(f"ast_{version}_{t}_mean")
            cols.append(f"ast_{version}_node_count_sum")
            cols.append(f"ast_{version}_node_count_mean")
        for version in ["before", "after"]:
            for t in self.cfg_types:
                cols.append(f"cfg_{version}_{t}_sum")
                cols.append(f"cfg_{version}_{t}_mean")
            cols.append(f"cfg_{version}_total_nodes_sum")
            cols.append(f"cfg_{version}_total_nodes_mean")
            cols.append(f"cfg_{version}_total_edges_sum")
            cols.append(f"cfg_{version}_total_edges_mean")
            cols.append(f"cfg_{version}_function_count")
            cols.append(f"cfg_{version}_avg_nodes_per_func")
            cols.append(f"cfg_{version}_avg_edges_per_func")
            cols.append(f"cfg_{version}_cyclomatic_complexity")
        return cols

    def _save_column_names(self):
        """保存列名说明到文件，方便查阅"""
        fpath = os.path.join(config.FEATURES_DIR, "feature_columns.txt")
        cols = self._get_column_order()
        with open(fpath, "w", encoding="utf-8") as f:
            for i, col in enumerate(cols):
                f.write(f"{i+1}. {col}\n")

    def _check_pr_has_files(self, repo_key, pr_id):
        """检查 PR 是否至少有一些 AST/CFG 文件"""
        paths = self._get_pr_base_path(repo_key, pr_id)
        for p in [paths["ast_before"], paths["ast_after"], paths["cfg_before"], paths["cfg_after"]]:
            if os.path.exists(p) and len([f for f in os.listdir(p) if f.endswith(".json")]) > 0:
                return True
        return False

    def _extract_pr_features(self, repo_key, pr_data):
        """提取单个 PR 的所有特征"""
        pr_id = pr_data["pr_id"]

        if not self._check_pr_has_files(repo_key, pr_id):
            self.stats["skipped_no_ast"] += 1
            return None

        base_feats = self._extract_base_features(pr_data)
        ast_feats = self._extract_ast_features(repo_key, pr_id)
        cfg_feats = self._extract_cfg_features(repo_key, pr_id)

        row = self._build_feature_row(pr_id, repo_key, base_feats, ast_feats, cfg_feats)
        return row

    def run_all(self, force_check: bool = False):
        """遍历所有 repo 的 pulls.json，提取特征写入 CSV
        
        Args:
            force_check: False=跳过所有检查直接返回, True=遍历提取特征
        """
        if not force_check:
            print("\n" + "=" * 60)
            print("FORCE_CHECK=False, 跳过特征提取")
            print("=" * 60)
            return

        self._collect_node_types()

        print("\n" + "=" * 60)
        print("开始提取特征...")
        print("=" * 60)

        repo_pulls_files = []
        for owner, repo in config.TARGET_REPOS:
            repo_key = f"{owner}_{repo}"
            pulls_file = os.path.join(config.HUMAN_DIR, f"{repo_key}_pulls.json")
            if os.path.exists(pulls_file):
                repo_pulls_files.append((repo_key, pulls_file))

        all_rows = []
        for repo_key, pulls_file in repo_pulls_files:
            print(f"\n处理 {repo_key}...")
            with open(pulls_file, "r", encoding="utf-8") as f:
                pulls = json.load(f)
            self.stats["total_pr"] += len(pulls)
            for pr in tqdm(pulls, desc=f"  {repo_key}"):
                row = self._extract_pr_features(repo_key, pr)
                if row is not None:
                    all_rows.append(row)
                    self.stats["success"] += 1
                else:
                    self.stats["failed"] += 1

        if not all_rows:
            print("\n没有提取到任何特征！")
            return None

        cols = self._get_column_order()
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        with open(self.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(all_rows)

        self._save_column_names()

        print("\n" + "=" * 60)
        print("特征提取完成！")
        print(f"  总 PR 数:   {self.stats['total_pr']}")
        print(f"  提取成功:   {self.stats['success']}")
        print(f"  跳过(无文件): {self.stats['skipped_no_ast']}")
        print(f"  提取失败:   {self.stats['failed']}")
        print(f"  特征维度:   {len(cols)}")
        print(f"  输出文件:   {self.output_csv}")
        print("=" * 60)

        return self.output_csv