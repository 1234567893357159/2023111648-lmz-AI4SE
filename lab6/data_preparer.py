"""
实验六 步骤一：数据准备模块
读取实验五中的 AI 生成代码数据，提取字段（含新增的 Issue 信息、Repository 上下文字段），
为后续上下文构建做准备。

遵循 lab1 风格：类 + 静态方法，函数逻辑在 .py 中，调用在 main.ipynb 中。
"""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab5"))

import config


class Lab6DataPreparer:
    """实验六数据准备器"""

    # ========================================================================
    #  数据加载
    # ========================================================================

    @staticmethod
    def load_ai_prs():
        """从 lab5 的 AI PR 目录加载所有 AI 生成代码的 PR 原始数据"""
        all_prs = []
        for owner, repo_name in config.TARGET_REPOS:
            repo_key = f"{owner}_{repo_name}"
            json_path = os.path.join(config.LAB5_AI_PULLS_DIR, f"{repo_key}_pulls.json")
            if not os.path.exists(json_path):
                print(f"  [跳过] 文件不存在: {json_path}")
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                prs = json.load(f)
            for pr in prs:
                pr["repo_key"] = repo_key
            all_prs.extend(prs)
            print(f"  加载 {repo_key}: {len(prs)} 个 AI PR")
        print(f"  共加载 {len(all_prs)} 个 AI PR")
        return all_prs

    @staticmethod
    def sample_prs(raw_prs, max_count=100, seed=42):
        """随机采样 PR，使用与 lab5 相同的种子保证公平对比"""
        random.seed(seed)
        if len(raw_prs) > max_count:
            sampled = random.sample(raw_prs, max_count)
        else:
            sampled = raw_prs[:]
        print(f"  采样完成: {len(sampled)}/{len(raw_prs)} 条 PR (seed={seed})")
        return sampled

    # ========================================================================
    #  字段提取（基础字段 — 与 lab5 兼容）
    # ========================================================================

    @staticmethod
    def extract_base_fields(pr):
        """提取与 lab5 一致的基础字段"""
        files = pr.get("files", [])
        commits = pr.get("commits", [])

        diff_parts = []
        for f_item in files:
            filename = f_item.get("filename", "")
            patch = f_item.get("patch", "")
            if patch:
                diff_parts.append(f"--- a/{filename}\n+++ b/{filename}\n{patch}")
        diff = "\n".join(diff_parts)

        commit_msgs = [c.get("message", "") for c in commits]
        commit_message = "\n---\n".join(commit_msgs)

        reviews = pr.get("reviews", [])
        review_comments = pr.get("review_comments", [])
        issue_comments = pr.get("issue_comments", [])

        file_names = [f_item.get("filename", "") for f_item in files]

        modified_functions = []
        for f_item in files:
            funcs = f_item.get("modified_functions", [])
            if isinstance(funcs, list):
                for fn in funcs:
                    if isinstance(fn, str) and fn.strip():
                        modified_functions.append(fn.strip())

        historical_parts = []
        for rc in review_comments:
            body = rc.get("body", "").strip()
            if body:
                historical_parts.append(
                    f"[Inline Review by {rc.get('user', 'unknown')}] {body}"
                )
        for ic in issue_comments:
            body = ic.get("body", "").strip()
            if body:
                historical_parts.append(
                    f"[Issue Comment by {ic.get('user', 'unknown')}] {body}"
                )
        historical_comments_text = "\n".join(historical_parts)

        return {
            "pr_id": pr["pr_id"],
            "repo": pr["repo_key"],
            "pr_title": pr.get("title", ""),
            "pr_description": pr.get("body", "") or "",
            "diff": diff,
            "commit_message": commit_message,
            "label": 1 if pr.get("merged") else 0,
            "files_changed": pr.get("changed_files", 0),
            "additions": pr.get("additions", 0),
            "deletions": pr.get("deletions", 0),
            "file_names": file_names,
            "modified_functions": modified_functions,
            "historical_comments_text": historical_comments_text,
            "reviews": [
                {
                    "reviewer": r.get("reviewer", ""),
                    "state": r.get("state", ""),
                    "body": r.get("body", ""),
                }
                for r in reviews
            ],
            "review_comments": [
                {
                    "user": rc.get("user", ""),
                    "body": rc.get("body", ""),
                    "path": rc.get("path", ""),
                }
                for rc in review_comments
            ],
            "issue_comments": [
                {
                    "user": ic.get("user", ""),
                    "body": ic.get("body", ""),
                }
                for ic in issue_comments
            ],
        }

    # ========================================================================
    #  字段提取（新增字段 — lab6 专属）
    # ========================================================================

    @staticmethod
    def extract_issue_references(pr):
        """从 PR body 和 issue_comments 中提取 Issue 引用编号"""
        body = pr.get("body", "") or ""
        references = []

        seen = set()
        for pattern in config.ISSUE_REFERENCE_PATTERNS:
            for match in re.finditer(pattern, body, re.IGNORECASE):
                issue_num = match.group(1)
                if issue_num not in seen:
                    references.append({
                        "issue_number": int(issue_num),
                        "source": "pr_body",
                        "context": body[max(0, match.start() - 50):match.end() + 50],
                    })
                    seen.add(issue_num)

        for ic in pr.get("issue_comments", []):
            comment_body = ic.get("body", "") or ""
            for pattern in config.ISSUE_REFERENCE_PATTERNS:
                for match in re.finditer(pattern, comment_body, re.IGNORECASE):
                    issue_num = match.group(1)
                    if issue_num not in seen:
                        references.append({
                            "issue_number": int(issue_num),
                            "source": "issue_comment",
                            "context": comment_body[max(0, match.start() - 50):match.end() + 50],
                        })
                        seen.add(issue_num)

        return references

    @staticmethod
    def detect_language(filename):
        """根据文件扩展名检测编程语言"""
        ext = os.path.splitext(filename)[1].lower()
        if ext in config.LANGUAGE_MAP:
            return config.LANGUAGE_MAP[ext]
        if filename.upper() == "BUILD" or filename.endswith(".BUILD"):
            return "Starlark"
        if filename.upper() == "MAKEFILE" or filename.endswith(".mk"):
            return "Makefile"
        return "Unknown"

    @staticmethod
    def extract_files_with_content(pr):
        """提取每个修改文件的详细信息，包括完整 patch 和语言"""
        files = pr.get("files", [])
        result = []
        for f in files:
            result.append({
                "filename": f.get("filename", ""),
                "status": f.get("status", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
                "patch": f.get("patch", ""),
                "modified_functions": f.get("modified_functions", []) if isinstance(f.get("modified_functions"), list) else [],
                "language": Lab6DataPreparer.detect_language(f.get("filename", "")),
                "raw_url": f.get("raw_url", ""),
            })
        return result

    @staticmethod
    def extract_related_functions(pr):
        """提取 diff 中引用的外部函数/类/模块（调用关系）"""
        imports_set = set()
        callees_set = set()

        for f in pr.get("files", []):
            patch = f.get("patch", "") or ""
            filename = f.get("filename", "")
            ext = os.path.splitext(filename)[1].lower()

            import_patterns = {
                ".py": r'(?:from|import)\s+([\w.]+)',
                ".go": r'import\s+[("]\s*([\w./]+)',
                ".java": r'import\s+([\w.]+)',
                ".ts": r'(?:from|import)\s+[\'"](.*?)[\'"]',
                ".tsx": r'(?:from|import)\s+[\'"](.*?)[\'"]',
                ".js": r'(?:require|from|import)\s*\(?[\'"](.*?)[\'"]',
                ".jsx": r'(?:require|from|import)\s*\(?[\'"](.*?)[\'"]',
            }

            pattern = import_patterns.get(ext)
            if pattern:
                for match in re.finditer(pattern, patch):
                    imports_set.add(match.group(1))

            modified = set()
            for mf in f.get("modified_functions", []):
                if isinstance(mf, str):
                    modified.add(mf)

            call_pattern = r'(?:^|[^\w])(\w+)\s*\('
            for match in re.finditer(call_pattern, patch, re.MULTILINE):
                name = match.group(1)
                if name not in modified and name.lower() not in config.FILTER_KEYWORDS:
                    callees_set.add(name)

        return {
            "imports": sorted(list(imports_set)),
            "callees": sorted(list(callees_set)),
        }

    # ========================================================================
    #  主流程
    # ========================================================================

    @staticmethod
    def extract_all_fields(pr):
        """提取单条 PR 的全部字段（基础 + 新增）"""
        base = Lab6DataPreparer.extract_base_fields(pr)
        base["issue_references"] = Lab6DataPreparer.extract_issue_references(pr)
        base["files_with_content"] = Lab6DataPreparer.extract_files_with_content(pr)
        base["related_functions"] = Lab6DataPreparer.extract_related_functions(pr)
        return base

    @staticmethod
    def compute_summary(data_list):
        """计算数据集的统计摘要"""
        repo_counts = {}
        merge_counts = {"merged": 0, "not_merged": 0}
        total_comments = 0
        issue_ref_counts = []

        for d in data_list:
            repo = d["repo"]
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
            if d["label"] == 1:
                merge_counts["merged"] += 1
            else:
                merge_counts["not_merged"] += 1
            total_comments += len(d.get("review_comments", [])) + len(d.get("issue_comments", []))
            issue_ref_counts.append(len(d.get("issue_references", [])))

        diff_lengths = [len(d["diff"]) for d in data_list]

        return {
            "total_prs": len(data_list),
            "merged": merge_counts["merged"],
            "not_merged": merge_counts["not_merged"],
            "repo_counts": repo_counts,
            "total_comments": total_comments,
            "avg_diff_length": sum(diff_lengths) / len(diff_lengths) if diff_lengths else 0,
            "prs_with_issue_refs": sum(1 for c in issue_ref_counts if c > 0),
            "avg_issue_refs": sum(issue_ref_counts) / len(issue_ref_counts) if issue_ref_counts else 0,
            "total_files": sum(len(d.get("files_with_content", [])) for d in data_list),
            "avg_related_functions": sum(
                len(d.get("related_functions", {}).get("callees", []))
                for d in data_list
            ) / len(data_list) if data_list else 0,
        }

    @staticmethod
    def save_results(data_list, summary):
        """保存提取结果到 JSON 文件"""
        with open(config.SELECTED_PRS_PATH, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"\n  数据已保存到: {config.SELECTED_PRS_PATH}")

        summary_path = config.SELECTED_PRS_PATH.replace(".json", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  摘要已保存到: {summary_path}")

    @staticmethod
    def run_all(max_prs=100, seed=42):
        """执行完整数据准备流程（供 main.ipynb 调用）"""
        print("=" * 60)
        print("  实验六 步骤一：数据准备")
        print("=" * 60)

        print("\n[1/4] 从 lab5 加载 AI PR 数据...")
        raw_prs = Lab6DataPreparer.load_ai_prs()

        print(f"\n[2/4] 随机采样 {max_prs} 条 PR (seed={seed})...")
        sampled_prs = Lab6DataPreparer.sample_prs(raw_prs, max_prs, seed)

        print("\n[3/4] 提取字段（基础字段 + Issue 信息 + Repository 上下文字段）...")
        extracted_prs = [Lab6DataPreparer.extract_all_fields(pr) for pr in sampled_prs]
        print(f"  提取完成，共 {len(extracted_prs)} 条")

        print("\n[4/4] 计算摘要并保存...")
        summary = Lab6DataPreparer.compute_summary(extracted_prs)
        Lab6DataPreparer.save_results(extracted_prs, summary)

        print(f"\n{'=' * 60}")
        print(f"  步骤一完成!")
        print(f"  总 PR 数:     {summary['total_prs']}")
        print(f"  已合并:       {summary['merged']}")
        print(f"  未合并:       {summary['not_merged']}")
        print(f"  含 Issue 引用: {summary['prs_with_issue_refs']} 条")
        print(f"  平均 Issue 引用: {summary['avg_issue_refs']:.1f} 个/PR")
        print(f"  总修改文件数: {summary['total_files']}")
        print(f"  平均调用关系: {summary['avg_related_functions']:.1f} 个/PR")
        print(f"{'=' * 60}")

        return extracted_prs, summary


if __name__ == "__main__":
    data, summary = Lab6DataPreparer.run_all()