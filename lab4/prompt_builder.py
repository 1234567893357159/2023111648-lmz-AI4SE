"""
实验四 阶段三：Prompt 设计模块
提供 4 种 Prompt 类型 x 2 个任务 = 8 个 Prompt 模板（英文版）
"""

PROMPT_TYPES = ["zero_shot", "few_shot", "cot", "role_based"]

# ========== Few-shot 示例 ==========
FEW_SHOT_EXAMPLE_MERGE_1 = """Example 1:
Code Change:
@@ -10,6 +10,6 @@ def calculate(a, b):
-    return a + b
+    if b == 0:
+        return 0
+    return a + b
Output:
{"decision": "Yes", "reason": "Added division-by-zero guard, improves robustness with clear logic"}"""

FEW_SHOT_EXAMPLE_MERGE_2 = """Example 2:
Code Change:
@@ -1,5 +1,15 @@ import os
+import subprocess
+
 def process(data):
-    result = data.strip()
+    result = subprocess.check_output(data, shell=True)
     return result
Output:
{"decision": "No", "reason": "Using shell=True introduces command injection risk without input validation"}"""

FEW_SHOT_EXAMPLE_REVIEW_1 = """Example 1:
Code Change:
@@ -10,7 +10,7 @@ def calculate(a, b):
-    return a + b
+    if b == 0:
+        return 0
+    return a + b
Review Comment: Consider extracting the zero-check logic into a separate helper function and adding unit tests for edge cases (b=0, negative values)."""

FEW_SHOT_EXAMPLE_REVIEW_2 = """Example 2:
Code Change:
@@ -20,6 +20,10 @@ class UserService:
     def get_user(self, user_id):
-        return self.db.query(user_id)
+        user = self.db.query(user_id)
+        if user is None:
+            return None
+        return user
Review Comment: Returning None from get_user may cause NPE at call sites. Consider throwing UserNotFoundException or returning Optional<User> for safer error handling."""


def build_prompt(context, task, prompt_type):
    """
    Build a prompt string.

    Args:
        context: Code context string
        task: "merge_prediction" or "review_comment"
        prompt_type: "zero_shot" / "few_shot" / "cot" / "role_based"

    Returns:
        str: Complete prompt
    """
    if task == "merge_prediction":
        return _build_merge_prompt(context, prompt_type)
    elif task == "review_comment":
        return _build_review_prompt(context, prompt_type)
    else:
        raise ValueError(f"Unknown task: {task}")


def _build_merge_prompt(context, prompt_type):
    if prompt_type == "zero_shot":
        return (
            "Determine whether the following Pull Request code change is likely to be merged.\n\n"
            "Output ONLY a JSON object, nothing else:\n"
            '{"decision": "Yes" or "No", "reason": "brief reason"}\n\n'
            f"Code Change:\n{context}"
        )

    elif prompt_type == "few_shot":
        return (
            "Determine whether the following Pull Request code change is likely to be merged.\n\n"
            "Here are two examples to guide your judgment and output format:\n\n"
            f"{FEW_SHOT_EXAMPLE_MERGE_1}\n\n"
            f"{FEW_SHOT_EXAMPLE_MERGE_2}\n\n"
            "---\n\n"
            "Now evaluate the following code change:\n\n"
            f"{context}\n\n"
            "Output ONLY a JSON object, nothing else:\n"
            '{"decision": "Yes" or "No", "reason": "brief reason"}'
        )

    elif prompt_type == "cot":
        return (
            "Analyze the following code change step by step, then determine whether the Pull Request is likely to be merged.\n\n"
            "Follow these steps:\n"
            "1. Purpose: What is the goal of this code change?\n"
            "2. Correctness: Is the logic correct? Are there potential bugs?\n"
            "3. Security: Are there security risks (injection, permission bypass, etc.)?\n"
            "4. Code Style: Does the code follow project conventions?\n"
            "5. Conclusion: Based on the above, should this PR be merged?\n\n"
            f"Code Change:\n{context}\n\n"
            "Provide your step-by-step analysis first, then output a JSON object:\n"
            '{"decision": "Yes" or "No", "reason": "summary reason"}'
        )

    elif prompt_type == "role_based":
        return (
            "You are a senior committer at the Apache Software Foundation with 10 years of experience "
            "reviewing large-scale open-source projects. Your review standards are extremely strict. "
            "Code must be:\n"
            "- Logically correct and bug-free\n"
            "- Consistent with project coding conventions\n"
            "- Well-tested with adequate coverage\n"
            "- Free of security vulnerabilities\n\n"
            "Review the following code change and determine if it passes review:\n\n"
            f"{context}\n\n"
            "Output ONLY a JSON object, nothing else:\n"
            '{"decision": "Yes" or "No", "reason": "brief reason"}'
        )

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


def _build_review_prompt(context, prompt_type):
    if prompt_type == "zero_shot":
        return (
            "You are a code review assistant. Write a review comment for the following Pull Request code change.\n\n"
            f"Code Change:\n{context}\n\n"
            "Output only the review comment, nothing else."
        )

    elif prompt_type == "few_shot":
        return (
            "You are a code review assistant. Write a review comment for the following code change.\n\n"
            "Here are two examples of good review comments for reference:\n\n"
            f"{FEW_SHOT_EXAMPLE_REVIEW_1}\n\n"
            f"{FEW_SHOT_EXAMPLE_REVIEW_2}\n\n"
            "---\n\n"
            "Now review the following code change:\n\n"
            f"{context}\n\n"
            "Output only the review comment, nothing else."
        )

    elif prompt_type == "cot":
        return (
            "Analyze the following code change step by step, then write a review comment.\n\n"
            "Follow these steps:\n"
            "1. Logic: Is the code logic correct? Are there potential bugs?\n"
            "2. Security: Are there any security risks?\n"
            "3. Style: Does the code follow conventions and best practices?\n"
            "4. Readability: Is the code easy to understand and maintain?\n"
            "5. Review Comment: Based on the above, provide a specific review comment.\n\n"
            f"Code Change:\n{context}\n\n"
            "Provide your step-by-step analysis first, then write a specific review comment."
        )

    elif prompt_type == "role_based":
        return (
            "You are a senior committer at the Apache Software Foundation with 10 years of experience "
            "reviewing large-scale open-source projects. Your review standards are extremely strict. "
            "Code must be:\n"
            "- Logically correct and bug-free\n"
            "- Consistent with project coding conventions\n"
            "- Well-tested with adequate coverage\n"
            "- Free of security vulnerabilities\n\n"
            "Review the following code change and provide a specific review comment:\n\n"
            f"{context}\n\n"
            "Output only the review comment, nothing else."
        )

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


def build_all_prompts(context, task):
    """Generate all 4 prompt types for a given context"""
    return {pt: build_prompt(context, task, pt) for pt in PROMPT_TYPES}


def print_prompt_examples(contexts):
    """Print sample prompts for notebook display"""
    if not contexts:
        print("No data")
        return

    c = contexts[0]
    ctx = c["contexts"]["diff_pr_desc"]

    print("=" * 60)
    print("Prompt Examples (Merge Prediction, diff_pr_desc context)")
    print("=" * 60)

    for pt in PROMPT_TYPES:
        prompt = build_prompt(ctx, "merge_prediction", pt)
        preview = prompt[:500].replace("\n", "\\n")
        print(f"\n--- {pt} ---")
        print(f"  Length: {len(prompt):,} chars")
        print(f"  Preview: {preview}...")

    print()
    print("=" * 60)
    print("Prompt Examples (Review Comment, diff_pr_desc context)")
    print("=" * 60)

    for pt in PROMPT_TYPES:
        prompt = build_prompt(ctx, "review_comment", pt)
        preview = prompt[:500].replace("\n", "\\n")
        print(f"\n--- {pt} ---")
        print(f"  Length: {len(prompt):,} chars")
        print(f"  Preview: {preview}...")