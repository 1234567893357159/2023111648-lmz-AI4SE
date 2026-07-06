"""
特征提取模块
从原始 PR 数据中提取结构化特征，包括：
- 基础统计特征（PR长度、文件数、Reviewer数等）
- AI Reviewer 检测
- AI 生成代码检测
"""

import re
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd

import config


class FeatureExtractor:
    """特征提取器"""

    @staticmethod
    def extract_basic_features(pr_data: Dict) -> Dict:
        """提取基础统计特征"""
        features = {
            "pr_id": pr_data.get("pr_id"),
            "repo_owner": pr_data.get("repo_owner"),
            "repo_name": pr_data.get("repo_name"),
            "title": pr_data.get("title"),
            "author": pr_data.get("author"),
            "created_at": pr_data.get("created_at"),

            # PR 长度（Body 的字符数）
            "pr_length": len(pr_data.get("body", "")),

            # PR 标题长度
            "title_length": len(pr_data.get("title", "")),

            # 修改文件数
            "files_changed": pr_data.get("changed_files", 0),

            # 代码增加行数
            "additions": pr_data.get("additions", 0),

            # 代码删除行数
            "deletions": pr_data.get("deletions", 0),

            # 是否 Merge
            "is_merged": 1 if pr_data.get("merged") else 0,

            # Label 数量
            "label_count": len(pr_data.get("labels", [])),

            # Label 列表（逗号分隔）
            "labels": ",".join(pr_data.get("labels", [])),

            # 修改函数相关信息（从 files 中提取）
            "modified_function_count": sum(
                len(f.get("modified_functions", []))
                for f in pr_data.get("files", [])
            ),
            "modified_functions": ",".join(list(dict.fromkeys(
                func
                for f in pr_data.get("files", [])
                for func in f.get("modified_functions", [])
            ))),
        }
        return features

    @staticmethod
    def extract_review_features(pr_data: Dict) -> Dict:
        """提取 Review 相关特征"""
        reviews = pr_data.get("reviews", [])
        issue_comments = pr_data.get("issue_comments", [])
        # 获取行内 Review Comments（针对具体代码行的评论）
        review_comments = pr_data.get("review_comments", [])

        # Reviewer 列表（去重）
        reviewers = list(set(
            r.get("reviewer") for r in reviews
            if r.get("reviewer")
        ))

        # Review 状态列表
        review_states = [r.get("state") for r in reviews if r.get("state")]

        # 计算各状态数量
        approved_count = review_states.count("APPROVED")
        changes_requested_count = review_states.count("CHANGES_REQUESTED")
        commented_count = review_states.count("COMMENTED")

        features = {
            # Reviewer 数量（去重）
            "reviewer_count": len(reviewers),

            # Reviewer 列表
            "reviewers": ",".join(reviewers),

            # 总 Review 次数
            "review_count": len(reviews),

            # Review Comment 数量（Reviews 中的评论）
            "review_comment_count": sum(
                1 for r in reviews if r.get("body") and r.get("body").strip()
            ),

            # Issue Comment 数量（PR 讨论评论）
            "issue_comment_count": len(issue_comments),

            # 行内 Review Comment 数量（针对具体代码行的评论）
            "inline_comment_count": len(review_comments),

            # 总评论数（Review Body + Issue Comments + 行内 Review Comments）
            "total_comment_count": (
                sum(1 for r in reviews if r.get("body") and r.get("body").strip())
                + len(issue_comments)
                + len(review_comments)
            ),

            # Review 状态统计
            "approved_count": approved_count,
            "changes_requested_count": changes_requested_count,
            "commented_count": commented_count,

            # 最终 Review Decision（取最后一个 Review 的状态）
            "final_review_decision": review_states[-1] if review_states else None,

            # 是否有 Review
            "has_review": 1 if len(reviews) > 0 else 0,
        }
        return features

    @staticmethod
    def detect_ai_reviewer(pr_data: Dict) -> Dict:
        """
        检测是否存在 AI Reviewer
        判断依据：
        1. Reviewer 用户名包含 bot 关键词
        2. Review Comment 内容模式匹配
        """
        reviews = pr_data.get("reviews", [])
        issue_comments = pr_data.get("issue_comments", [])

        ai_reviewers_found = []
        has_ai_reviewer = 0

        # 检查每个 Review 的 Reviewer
        for review in reviews:
            reviewer = review.get("reviewer", "")
            if reviewer:
                # 检查用户名是否包含 AI/Bot 关键词
                reviewer_lower = reviewer.lower()
                for keyword in config.AI_REVIEWER_KEYWORDS:
                    if keyword in reviewer_lower:
                        ai_reviewers_found.append(reviewer)
                        has_ai_reviewer = 1
                        break

        # 检查 Issue Comments 中是否有 AI 评论
        for comment in issue_comments:
            user = comment.get("user", "")
            if user:
                user_lower = user.lower()
                for keyword in config.AI_REVIEWER_KEYWORDS:
                    if keyword in user_lower:
                        if user not in ai_reviewers_found:
                            ai_reviewers_found.append(user)
                            has_ai_reviewer = 1
                        break

        features = {
            "has_ai_reviewer": has_ai_reviewer,
            "ai_reviewers": ",".join(ai_reviewers_found),
            "ai_reviewer_count": len(ai_reviewers_found),
        }
        return features

    @staticmethod
    def detect_ai_generated_code(pr_data: Dict) -> Dict:
        """
        检测是否包含 AI 生成代码（二分类：是 AI / 不是 AI）

        判断策略：
        1. Bot 作者 → 直接排除，不是 AI
        2. commit message / PR body / PR title / labels 中匹配 AI 关键词 → 是 AI
        3. 启发式规则：单文件大新增 + 短 commit / PR body 模板化 → 是 AI
        """
        commits = pr_data.get("commits", [])
        body = (pr_data.get("body") or "").lower()
        title = (pr_data.get("title") or "").lower()
        author = (pr_data.get("author") or "").lower()
        labels = [lbl.lower() for lbl in pr_data.get("labels", [])]

        has_ai_code = 0
        ai_indicators = []

        # ====== 第一步：Bot 作者直接排除 ======
        if any(kw in author for kw in config.BOT_AUTHOR_KEYWORDS):
            return {
                "has_ai_generated_code": 0,
                "ai_code_indicators": "bot_excluded",
            }

        # ====== 第二步：关键词匹配 ======
        # 汇总所有可搜索文本
        all_commit_text = " ".join(
            c.get("message", "") for c in commits
        ).lower()
        search_texts = {
            "commit": all_commit_text,
            "body": body,
            "title": title,
            "labels": " ".join(labels),
        }

        for source, text in search_texts.items():
            if not text:
                continue
            for keyword in config.AI_KEYWORDS:
                if keyword in text:
                    has_ai_code = 1
                    ai_indicators.append(f"{source}: {keyword}")
                    break

        # ====== 第三步：启发式规则（关键词未命中时） ======
        if has_ai_code == 0:
            files_changed = pr_data.get("changed_files", 0)
            additions = pr_data.get("additions", 0)
            first_commit = commits[0].get("message", "") if commits else ""

            # 规则1：单文件大新增 + 极短 commit → AI 特征
            if files_changed <= 2 and additions >= 200:
                if len(first_commit.strip()) <= 30:
                    has_ai_code = 1
                    ai_indicators.append(
                        "heuristic: large_single_file_short_commit"
                    )

            # 规则2：PR body 高度模板化 → AI 特征
            if has_ai_code == 0:
                template_markers = [
                    "## summary", "## changes", "## testing",
                    "## description", "## motivation",
                ]
                match_count = sum(1 for m in template_markers if m in body)
                if match_count >= 3 and len(body) > 500:
                    has_ai_code = 1
                    ai_indicators.append("heuristic: template_body")

        return {
            "has_ai_generated_code": has_ai_code,
            "ai_code_indicators": "; ".join(ai_indicators) if ai_indicators else "",
        }

    @staticmethod
    def extract_language_features(pr_data: Dict) -> Dict:
        """
        提取编程语言统计特征
        统计 PR 修改的文件使用了哪些编程语言
        """
        files = pr_data.get("files", [])
        
        # 常见编程语言扩展名映射
        EXT_TO_LANG = {
            # Go
            '.go': 'Go',
            # Python
            '.py': 'Python', '.pyi': 'Python',
            # JavaScript/TypeScript
            '.js': 'JavaScript', '.jsx': 'JavaScript',
            '.ts': 'TypeScript', '.tsx': 'TypeScript',
            # Java
            '.java': 'Java',
            # C/C++
            '.c': 'C', '.h': 'C',
            '.cpp': 'C++', '.cc': 'C++', '.cxx': 'C++',
            '.hpp': 'C++', '.cppm': 'C++',
            # C#
            '.cs': 'C#',
            # Rust
            '.rs': 'Rust',
            # Ruby
            '.rb': 'Ruby',
            # PHP
            '.php': 'PHP',
            # Swift
            '.swift': 'Swift',
            # Kotlin
            '.kt': 'Kotlin', '.kts': 'Kotlin',
            # Scala
            '.scala': 'Scala',
            # Haskell
            '.hs': 'Haskell',
            # Lua
            '.lua': 'Lua',
            # Shell
            '.sh': 'Shell', '.bash': 'Shell', '.zsh': 'Shell',
            # PowerShell
            '.ps1': 'PowerShell',
            # HTML/CSS
            '.html': 'HTML', '.htm': 'HTML',
            '.css': 'CSS', '.scss': 'CSS', '.less': 'CSS', '.sass': 'CSS',
            # JSON/YAML/XML
            '.json': 'JSON', '.jsonc': 'JSON',
            '.yaml': 'YAML', '.yml': 'YAML',
            '.xml': 'XML',
            # Markdown/Docs
            '.md': 'Markdown', '.markdown': 'Markdown',
            '.rst': 'RST', '.txt': 'Text',
            # SQL
            '.sql': 'SQL',
            # Docker
            'dockerfile': 'Dockerfile',
            # Makefile/Build
            '.makefile': 'Makefile', 'makefile': 'Makefile',
            'makefile': 'Makefile', 'Makefile': 'Makefile',
            '.cmake': 'CMake', 'CMakeLists.txt': 'CMake',
            # Config
            '.toml': 'TOML', '.ini': 'INI', '.cfg': 'INI',
            '.gitignore': 'GitIgnore', '.dockerignore': 'DockerIgnore',
        }
        
        # 统计每种语言的文件数量和总行数
        language_counts = {}
        language_additions = {}
        language_deletions = {}
        
        for file in files:
            filename = file.get("filename", "").lower()
            additions = file.get("additions", 0)
            deletions = file.get("deletions", 0)
            
            # 获取文件扩展名
            if filename.endswith('dockerfile') or filename == 'dockerfile':
                lang = 'Dockerfile'
            elif 'cmakelists.txt' in filename:
                lang = 'CMake'
            else:
                ext = '.' + filename.split('.')[-1] if '.' in filename else ''
                lang = EXT_TO_LANG.get(ext, None)
            
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
                language_additions[lang] = language_additions.get(lang, 0) + additions
                language_deletions[lang] = language_deletions.get(lang, 0) + deletions
            elif ext:
                # 未分类的扩展名将其归类为 Other
                lang = f'Other({ext[1:]})'
                language_counts[lang] = language_counts.get(lang, 0) + 1
                language_additions[lang] = language_additions.get(lang, 0) + additions
                language_deletions[lang] = language_deletions.get(lang, 0) + deletions
        
        # 找出占比最大的编程语言（按修改文件数量）
        if language_counts:
            primary_language = max(language_counts.items(), key=lambda x: x[1])[0]
            total_languages = len(language_counts)
            languages = ','.join(language_counts.keys())
        else:
            primary_language = None
            total_languages = 0
            languages = ""
        
        features = {
            "primary_language": primary_language,
            "total_languages": total_languages,
            "languages": languages,
        }
        
        return features
    
    @staticmethod
    def extract_all_features(pr_data: Dict) -> Dict:
        """提取所有特征"""
        features = {}
        features.update(FeatureExtractor.extract_basic_features(pr_data))
        features.update(FeatureExtractor.extract_review_features(pr_data))
        features.update(FeatureExtractor.extract_language_features(pr_data))
        features.update(FeatureExtractor.detect_ai_reviewer(pr_data))
        features.update(FeatureExtractor.detect_ai_generated_code(pr_data))
        return features

    @staticmethod
    def create_dataframe(all_prs: List[Dict]) -> pd.DataFrame:
        """从所有 PR 数据创建 DataFrame"""
        all_features = []
        for pr in all_prs:
            features = FeatureExtractor.extract_all_features(pr)
            all_features.append(features)

        df = pd.DataFrame(all_features)
        return df


def build_dataset() -> pd.DataFrame:
    """
    从原始数据构建完整的数据集
    遍历 RAW_DIR 中的 JSON 文件，提取特征并保存为 CSV
    """
    from github_api import get_all_prs_data

    print("=" * 60)
    print("开始构建数据集...")
    print("=" * 60)

    # 加载所有原始数据
    all_pulls = get_all_prs_data()
    print(f"共加载 {len(all_pulls)} 个 PR")

    # 提取特征
    df = FeatureExtractor.create_dataframe(all_pulls)
    print(f"数据集大小: {df.shape[0]} 行, {df.shape[1]} 列")

    # 保存 CSV
    csv_file = os.path.join(config.PROCESSED_DIR, "dataset.csv")
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"数据集已保存到: {csv_file}")

    # 保存统计摘要
    summary = df.describe(include='all')
    summary_file = os.path.join(config.PROCESSED_DIR, "summary.csv")
    summary.to_csv(summary_file, encoding='utf-8-sig')
    print(f"统计摘要已保存到: {summary_file}")

    return df


if __name__ == "__main__":
    # 测试
    df = build_dataset()
    print("\n数据预览:")
    print(df.head())
    print(f"\n数据集列: {list(df.columns)}")