"""
步骤 5a v2：按文件粒度拆分代码审查意见生成数据
改进：每个样本 = 一个文件的 diff + 对应文件的 review comments
解决原方案中整个 PR diff 过长导致截断丢失信息的问题
"""

import json
import os
import re

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config


AUTO_GENERATED_PATTERNS = [
    r"(^|/)package-lock\.json$",
    r"(^|/)yarn\.lock$",
    r"(^|/)pnpm-lock\.yaml$",
    r"(^|/)go\.sum$",
    r"(^|/)Cargo\.lock$",
    r"(^|/)Gemfile\.lock$",
    r"(^|/)poetry\.lock$",
    r"(^|/)Pipfile\.lock$",
    r"\.pb\.go$",
    r"\.pb\.cc$",
    r"\.pb\.h$",
    r"\.pb\.[a-z]+$",
    r"\.generated\.\w+$",
    r"\.auto\.\w+$",
    r"\.min\.js$",
    r"\.min\.css$",
    r"\.pyc$",
    r"\.pyo$",
    r"(^|/)vendor/",
    r"(^|/)node_modules/",
    r"(^|/)__pycache__/",
    r"\.patch$",
    r"\.diff$",
    r"\.lock$",
]


def is_auto_generated(filename: str) -> bool:
    for pattern in AUTO_GENERATED_PATTERNS:
        if re.search(pattern, filename):
            return True
    return False


def compress_patch(patch: str, max_chars: int = 1200) -> str:
    """精简 unified diff patch：去掉上下文行，只保留变更行和 hunk 头"""
    lines = patch.split("\n")
    compressed = []
    context_count = 0

    for line in lines:
        if line.startswith("@@"):
            if context_count > 0:
                compressed.append(f"... ({context_count} context lines omitted)")
                context_count = 0
            compressed.append(line)
        elif line.startswith("+") or line.startswith("-"):
            if context_count > 0:
                compressed.append(f"... ({context_count} context lines omitted)")
                context_count = 0
            compressed.append(line)
        elif line.startswith(" ") or line == "":
            context_count += 1
        else:
            if context_count > 0:
                compressed.append(f"... ({context_count} context lines omitted)")
                context_count = 0
            compressed.append(line)

    if context_count > 0:
        compressed.append(f"... ({context_count} context lines omitted)")

    result = "\n".join(compressed)

    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"

    return result


class CodeReviewDataPreparerV2:
    """按文件粒度拆分的代码审查数据准备器"""

    def __init__(self):
        self._pr_cache = {}

    def _load_lab1_prs(self, repo_key: str) -> dict:
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
        print(f"  加载 {repo_key}: {len(pr_list)} PRs")
        return index

    def _build_input_for_file(self, pr: dict, file_info: dict) -> str:
        """构建单个文件的输入：PR 上下文 + 该文件的 patch"""
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
            commit_msgs = [m[:200] for m in commit_msgs]
            parts.append(f"commits: {' | '.join(commit_msgs)}")

        filename = file_info.get("filename", "")
        status = file_info.get("status", "")
        additions = file_info.get("additions", 0)
        deletions = file_info.get("deletions", 0)
        parts.append(f"file: {filename} (status: {status}, +{additions} -{deletions})")

        patch = file_info.get("patch", "")
        if patch:
            compressed = compress_patch(patch)
            parts.append(f"patch:\n{compressed}")

        return "\n".join(parts)

    def _get_file_comments(self, pr: dict, filename: str) -> list:
        """获取与特定文件关联的 review comments"""
        review_comments = pr.get("review_comments", [])
        file_comments = []
        for rc in review_comments:
            rc_path = rc.get("path", "")
            if rc_path == filename:
                body = rc.get("body", "").strip()
                if body and len(body) > 3:
                    file_comments.append(body)
        return file_comments

    def _get_pr_reviews(self, pr: dict) -> list:
        """获取 PR 级别的 reviews（非文件特定）"""
        reviews = pr.get("reviews", [])
        pr_reviews = []
        for rv in reviews:
            body = rv.get("body", "").strip()
            if body and len(body) > 3:
                pr_reviews.append(body)
        return pr_reviews

    def prepare_split(self, split_path: str, output_path: str, desc: str) -> None:
        """处理一个划分，按文件拆分"""
        print(f"\n处理 {desc} 数据...")

        with open(split_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        results = []
        skipped_no_comments = 0
        skipped_no_patch = 0
        skipped_auto_generated = 0
        missing = 0

        for sample in samples:
            pr_id = sample["pr_id"]
            repo = sample["repo"]

            pr_index = self._load_lab1_prs(repo)
            pr = pr_index.get(pr_id)

            if pr is None:
                missing += 1
                continue

            files = pr.get("files", [])
            pr_reviews = self._get_pr_reviews(pr)

            for file_info in files:
                filename = file_info.get("filename", "")
                patch = file_info.get("patch", "")

                if not patch:
                    skipped_no_patch += 1
                    continue

                if is_auto_generated(filename):
                    skipped_auto_generated += 1
                    continue

                file_comments = self._get_file_comments(pr, filename)

                target_parts = file_comments + pr_reviews
                if not target_parts:
                    skipped_no_comments += 1
                    continue

                input_text = self._build_input_for_file(pr, file_info)
                target_text = "\n\n".join(target_parts)

                results.append({
                    "pr_id": pr_id,
                    "repo": repo,
                    "filename": filename,
                    "input": input_text,
                    "target": target_text,
                })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  {desc}: {len(results)} 样本, 跳过无评论: {skipped_no_comments}, 跳过无patch: {skipped_no_patch}, 跳过自动生成: {skipped_auto_generated}, 缺失 PR: {missing}")

    def run(self) -> None:
        """运行完整数据准备流程"""
        print("=" * 70)
        print("代码审查意见生成数据准备 v2：按文件粒度拆分")
        print("=" * 70)

        self.prepare_split(config.TRAIN_JSON_PATH, config.CODEREVIEW_TRAIN_PATH, "训练集")
        self.prepare_split(config.VAL_JSON_PATH, config.CODEREVIEW_VAL_PATH, "验证集")
        self.prepare_split(config.TEST_JSON_PATH, config.CODEREVIEW_TEST_PATH, "测试集")

        print("\n" + "=" * 70)
        print("数据准备完成")
        print(f"  训练集: {config.CODEREVIEW_TRAIN_PATH}")
        print(f"  验证集: {config.CODEREVIEW_VAL_PATH}")
        print(f"  测试集: {config.CODEREVIEW_TEST_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    preparer = CodeReviewDataPreparerV2()
    preparer.run()