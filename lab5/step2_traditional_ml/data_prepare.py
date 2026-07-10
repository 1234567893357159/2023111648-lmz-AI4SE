"""
步骤二：传统机器学习模型测试 - 数据准备
1. 从 lab1 原始数据中提取 AI 生成代码 PR 的完整信息
2. 下载 before/after 源代码（复用 lab2 的 CodeExtractor）
3. 生成 AST 和 CFG（复用 lab2 的 ASTCFGGenerator）
4. 提取特征并对齐列顺序，输出 ai_features.csv
"""

import json
import os
import sys
import csv
import time
from typing import List, Dict, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "lab2"))

import config
import pandas as pd
from tqdm import tqdm

from code_extractor import CodeExtractor
from ast_cfg_generator import ASTCFGGenerator


class AIDataPreparer:
    """AI 代码数据准备器"""

    def __init__(self):
        self.ai_pr_df = None
        self.standard_columns = []
        self.stats = {
            "step1_filter": 0,
            "total_ai_prs": 0,
            "code_downloaded": 0,
            "code_skipped": 0,
            "code_failed": 0,
            "ast_generated": 0,
            "ast_skipped": 0,
            "ast_failed": 0,
            "features_success": 0,
            "features_skipped": 0,
        }

    def load_standard_columns(self) -> List[str]:
        """加载 lab2 提取的标准特征列名，保证对齐"""
        cols = []
        with open(config.LAB2_FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(". ", 1)
                if len(parts) == 2:
                    cols.append(parts[1])
        self.standard_columns = cols
        print(f"加载 lab2 标准特征列: {len(cols)} 列")
        return cols

    def filter_ai_pulls(self) -> Dict:
        """步骤 A：从 lab1 原始数据中筛选 AI 生成代码 PR"""
        print("\n" + "=" * 60)
        print("步骤 A: 筛选 AI 生成代码 PR")
        print("=" * 60)

        # 读取步骤一筛选好的 PR 列表
        ai_pr_df = pd.read_csv(config.AI_PRS_PATH)
        self.ai_pr_df = ai_pr_df
        print(f"从步骤一获取 AI PR 列表: {len(ai_pr_df)} 条")

        # 按仓库分组
        repo_groups = {}
        for _, row in ai_pr_df.iterrows():
            owner = row["repo_owner"]
            repo_name = row["repo_name"]
            key = f"{owner}_{repo_name}"
            if key not in repo_groups:
                repo_groups[key] = []
            repo_groups[key].append(row["pr_id"])

        print(f"涉及 {len(repo_groups)} 个仓库: {list(repo_groups.keys())}")

        # 从 lab1 raw 文件中提取完整 PR 数据
        for repo_key, ai_pr_ids in repo_groups.items():
            owner, repo = repo_key.split("_")
            raw_file = os.path.join(config.LAB1_RAW_DIR, f"{repo_key}_pulls.json")

            if not os.path.exists(raw_file):
                print(f"  警告: 原始文件不存在 {raw_file}")
                continue

            print(f"  处理 {repo_key}...")
            with open(raw_file, "r", encoding="utf-8") as f:
                all_pulls = json.load(f)

            # 筛选出 AI PR
            ai_pulls = [pr for pr in all_pulls if pr.get("pr_id") in ai_pr_ids]
            print(f"    从 {len(all_pulls)} 个 PR 中筛选出 {len(ai_pulls)} 个 AI PR")

            # 保存
            output_file = os.path.join(config.AI_PULLS_DIR, f"{repo_key}_pulls.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ai_pulls, f, ensure_ascii=False, indent=2)

            self.stats["step1_filter"] += len(ai_pulls)

        print(f"\n筛选完成: 共 {self.stats['step1_filter']} 个 AI PR")
        return repo_groups

    def download_code(self, force_check: bool = False):
        """步骤 B: 下载 AI PR 的 before/after 源代码"""
        print("\n" + "=" * 60)
        print("步骤 B: 下载 AI PR 源代码")
        print("=" * 60)

        if not force_check:
            print("FORCE_CHECK=False, 跳过下载")
            return

        # lab2 的 code_extractor 导入 config 时拿到的是 lab5 的 config（Python 模块缓存），
        # 所以直接在 config 上设置 lab2 需要的字段即可，无需 monkey-patch
        config.HUMAN_CODE_DIR = config.AI_CODE_DIR
        config.HUMAN_DIR = config.AI_PULLS_DIR

        from code_extractor import CodeExtractor
        extractor = CodeExtractor()

        repo_files = []
        for owner, repo_name in config.TARGET_REPOS:
            repo_key = f"{owner}_{repo_name}"
            pulls_file = os.path.join(config.AI_PULLS_DIR, f"{repo_key}_pulls.json")
            if os.path.exists(pulls_file):
                with open(pulls_file, "r", encoding="utf-8") as f:
                    pulls = json.load(f)
                if pulls:
                    repo_files.append((owner, repo_name, pulls))

        if not repo_files:
            print("没有找到任何 AI PR 文件，请先运行 filter_ai_pulls()")
            return

        all_results = {}
        for owner, repo_name, pulls in repo_files:
            print(f"\n处理 {owner}/{repo_name}...")
            repo_key = f"{owner}_{repo_name}"
            results = []
            pbar = tqdm(pulls, desc=f"  {owner}/{repo_name}", unit="pr", ncols=100, ascii=True)
            for pr in pbar:
                result = extractor.extract_pr_code(owner, repo_name, pr, force_check=force_check)
                results.append(result)
                pbar.set_postfix({
                    "下载": extractor.stats["downloaded"],
                    "跳过": extractor.stats["skipped"],
                    "失败": extractor.stats["failed"],
                })
                time.sleep(config.REQUEST_DELAY)
            pbar.close()
            all_results[repo_key] = results

        self.stats["code_downloaded"] = extractor.stats["downloaded"]
        self.stats["code_skipped"] = extractor.stats["skipped"]
        self.stats["code_failed"] = extractor.stats["failed"]

        print(f"\n{'=' * 60}")
        print(f"代码下载完成!")
        print(f"  下载成功: {self.stats['code_downloaded']}")
        print(f"  跳过(已存在): {self.stats['code_skipped']}")
        print(f"  下载失败: {self.stats['code_failed']}")
        print(f"{'=' * 60}")

        return all_results

    def generate_ast_cfg(self, force_check: bool = False):
        """步骤 C: 生成 AST 和 CFG"""
        print("\n" + "=" * 60)
        print("步骤 C: 生成 AST 和 CFG")
        print("=" * 60)

        if not force_check:
            print("FORCE_CHECK=False, 跳过 AST/CFG 生成")
            return

        # lab2 的 ast_cfg_generator 导入 config 时拿到的是 lab5 的 config（Python 模块缓存），
        # 所以直接在 config 上设置 lab2 需要的字段即可
        config.HUMAN_AST_DIR = config.AI_AST_DIR
        config.HUMAN_CFG_DIR = config.AI_CFG_DIR
        config.HUMAN_CODE_DIR = config.AI_CODE_DIR

        from ast_cfg_generator import ASTCFGGenerator
        generator = ASTCFGGenerator()
        files = generator._collect_code_files()
        generator.stats["total"] = len(files)

        print(f"找到 {len(files)} 个代码文件需要处理")

        if len(files) == 0:
            print("没有找到代码文件，请先运行 download_code()")
            return

        pbar = tqdm(files, desc="  生成 AST/CFG", unit="file", ncols=100, ascii=True)
        for fpath, rel in pbar:
            generator._process_file(fpath, rel)
            pbar.set_postfix({
                "成功": generator.stats["success"],
                "跳过": generator.stats["skipped"],
                "失败": generator.stats["failed"],
            })
        pbar.close()

        self.stats["ast_generated"] = generator.stats["success"]
        self.stats["ast_skipped"] = generator.stats["skipped"]
        self.stats["ast_failed"] = generator.stats["failed"]

        print(f"\n{'=' * 60}")
        print(f"AST 和 CFG 生成完成!")
        print(f"  总文件数:     {generator.stats['total']}")
        print(f"  生成成功:     {self.stats['ast_generated']}")
        print(f"  跳过(已存在): {self.stats['ast_skipped']}")
        print(f"  生成失败:     {self.stats['ast_failed']}")
        print(f"  不支持语言:   {generator.stats['unsupported']}")
        print(f"{'=' * 60}")

    def _get_pr_base_path(self, repo_key, pr_id):
        """获取 PR 各种目录的基础路径"""
        return {
            "code_before": os.path.join(config.AI_CODE_DIR, repo_key, str(pr_id), "before"),
            "code_after": os.path.join(config.AI_CODE_DIR, repo_key, str(pr_id), "after"),
            "ast_before": os.path.join(config.AI_AST_DIR, repo_key, str(pr_id), "before"),
            "ast_after": os.path.join(config.AI_AST_DIR, repo_key, str(pr_id), "before"),
            "cfg_before": os.path.join(config.AI_CFG_DIR, repo_key, str(pr_id), "before"),
            "cfg_after": os.path.join(config.AI_CFG_DIR, repo_key, str(pr_id), "after"),
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

    def _extract_ast_from_dir(self, dir_path, ast_types):
        """从一个目录（before/after）提取 AST 特征"""
        counts = {t: 0 for t in ast_types}
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

    def _extract_ast_features(self, repo_key, pr_id, ast_types):
        """提取 before 和 after AST 特征"""
        paths = self._get_pr_base_path(repo_key, pr_id)

        results = {}
        for version, dir_path in [("before", paths["ast_before"]), ("after", paths["ast_after"])]:
            counts, total_nodes, file_count = self._extract_ast_from_dir(dir_path, ast_types)
            divide_by = max(file_count, 1)
            for t in ast_types:
                results[f"ast_{version}_{t}_sum"] = counts[t]
                results[f"ast_{version}_{t}_mean"] = counts[t] / divide_by
            results[f"ast_{version}_node_count_sum"] = total_nodes
            results[f"ast_{version}_node_count_mean"] = total_nodes / divide_by

        return results

    def _extract_cfg_from_dir(self, dir_path, cfg_types):
        """从一个目录（before/after）提取 CFG 特征"""
        type_counts = {t: 0 for t in cfg_types}
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

    def _extract_cfg_features(self, repo_key, pr_id, cfg_types):
        """提取 before 和 after CFG 特征"""
        paths = self._get_pr_base_path(repo_key, pr_id)

        results = {}
        for version, dir_path in [("before", paths["cfg_before"]), ("after", paths["cfg_after"])]:
            type_counts, total_nodes, total_edges, func_count = self._extract_cfg_from_dir(dir_path, cfg_types)
            divide_by = max(func_count, 1)
            for t in cfg_types:
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

    def _parse_standard_columns(self) -> tuple:
        """从标准列名中解析 AST 和 CFG 类型列表"""
        ast_types: Set[str] = set()
        cfg_types: Set[str] = set()

        for col in self.standard_columns:
            if col.startswith("ast_before_") and col.endswith("_sum"):
                parts = col.split("_")
                if len(parts) >= 4:
                    ast_type = "_".join(parts[2:-1])
                    ast_types.add(ast_type)
            elif col.startswith("cfg_before_") and col.endswith("_sum"):
                parts = col.split("_")
                if len(parts) >= 4:
                    cfg_type = "_".join(parts[2:-1])
                    cfg_types.add(cfg_type)

        return sorted(ast_types), sorted(cfg_types)

    def _check_pr_has_files(self, repo_key, pr_id):
        """检查 PR 是否至少有一些 AST/CFG 文件"""
        paths = self._get_pr_base_path(repo_key, pr_id)
        for p in [paths["ast_before"], paths["ast_after"], paths["cfg_before"], paths["cfg_after"]]:
            if os.path.exists(p) and len([f for f in os.listdir(p) if f.endswith(".json")]) > 0:
                return True
        return False

    def _align_columns(self, raw_row) -> Dict:
        """对齐到标准列顺序，缺失填 0"""
        aligned = {}
        for col in self.standard_columns:
            aligned[col] = raw_row.get(col, 0)
        return aligned

    def extract_features(self):
        """步骤 D: 提取特征并对齐列顺序"""
        print("\n" + "=" * 60)
        print("步骤 D: 提取特征")
        print("=" * 60)

        if not self.standard_columns:
            self.load_standard_columns()

        ast_types, cfg_types = self._parse_standard_columns()
        print(f"解析标准列: AST 类型 {len(ast_types)} 种, CFG 类型 {len(cfg_types)} 种")

        # 收集所有 PR
        all_pulls = []
        repo_files = []
        for owner, repo_name in config.TARGET_REPOS:
            repo_key = f"{owner}_{repo_name}"
            pulls_file = os.path.join(config.AI_PULLS_DIR, f"{repo_key}_pulls.json")
            if os.path.exists(pulls_file):
                with open(pulls_file, "r", encoding="utf-8") as f:
                    pulls = json.load(f)
                repo_files.append((repo_key, pulls))

        if not repo_files:
            print("没有找到任何 AI PR 文件")
            return None

        all_rows = []
        stats = {"total": 0, "success": 0, "skipped_no_ast": 0, "failed": 0}

        for repo_key, pulls in repo_files:
            print(f"\n处理 {repo_key}...")
            stats["total"] += len(pulls)
            for pr in tqdm(pulls, desc=f"  {repo_key}", unit="pr"):
                pr_id = pr["pr_id"]

                if not self._check_pr_has_files(repo_key, pr_id):
                    stats["skipped_no_ast"] += 1
                    continue

                base_feats = self._extract_base_features(pr)
                ast_feats = self._extract_ast_features(repo_key, pr_id, ast_types)
                cfg_feats = self._extract_cfg_features(repo_key, pr_id, cfg_types)

                raw_row = {
                    "pr_id": pr_id,
                    "repo": repo_key,
                }
                raw_row.update(base_feats)
                raw_row.update(ast_feats)
                raw_row.update(cfg_feats)

                aligned_row = self._align_columns(raw_row)
                all_rows.append(aligned_row)
                stats["success"] += 1

        self.stats["features_success"] = stats["success"]
        self.stats["features_skipped"] = stats["skipped_no_ast"]

        print(f"\n特征提取完成:")
        print(f"  总 PR: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  跳过(无文件): {stats['skipped_no_ast']}")
        print(f"  失败: {stats['failed']}")

        if not all_rows:
            print("没有提取到任何特征!")
            return None

        # 保存 CSV
        os.makedirs(os.path.dirname(config.AI_FEATURES_PATH), exist_ok=True)
        with open(config.AI_FEATURES_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.standard_columns)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\n特征文件已保存: {config.AI_FEATURES_PATH}")
        print(f"  特征维度: {len(self.standard_columns)}")

        return config.AI_FEATURES_PATH

    def run_all(self, force_code: bool = False, force_ast: bool = False):
        """执行完整流程"""
        print("=" * 60)
        print("步骤二数据准备: AI 生成代码 -> 特征提取")
        print("=" * 60)

        self.load_standard_columns()
        self.filter_ai_pulls()
        self.download_code(force_check=force_code)
        self.generate_ast_cfg(force_check=force_ast)
        self.extract_features()

        print("\n" + "=" * 60)
        print("数据准备完成!")
        print(f"  筛选出 AI PR: {self.stats['step1_filter']}")
        print(f"  下载代码文件: {self.stats['code_downloaded']} (跳过 {self.stats['code_skipped']}, 失败 {self.stats['code_failed']})")
        print(f"  生成 AST/CFG: {self.stats['ast_generated']} (跳过 {self.stats['ast_skipped']}, 失败 {self.stats['ast_failed']})")
        print(f"  提取特征成功: {self.stats['features_success']} (跳过 {self.stats['features_skipped']})")
        print("=" * 60)

        return self.stats


def main(force_code=False, force_ast=False):
    preparer = AIDataPreparer()
    stats = preparer.run_all(force_code=force_code, force_ast=force_ast)
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-code", action="store_true", help="强制检查下载代码")
    parser.add_argument("--force-ast", action="store_true", help="强制生成 AST/CFG")
    args = parser.parse_args()
    main(force_code=args.force_code, force_ast=args.force_ast)