"""
实验六 步骤三：Prompt 设计模块
提供 4 种 Prompt 类型 x 2 个任务 = 8 个 Prompt 模板
  - role_based:     角色扮演 Prompt
  - few_shot:       少样本示例 Prompt
  - cot:            思维链 Prompt（优化解析格式）
  - self_reflection: 自我反思 Prompt（单次调用两阶段）
"""

PROMPT_TYPES = ["role_based", "few_shot", "cot", "self_reflection"]

# ========== Few-shot 示例（复用 lab4）==========

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


# ========================================================================
#  统一入口
# ========================================================================

def build_prompt(context, task, prompt_type):
    """
    构建 Prompt 字符串。

    参数:
        context:     代码上下文文本
        task:        "merge_prediction" 或 "review_comment"
        prompt_type: "role_based" / "few_shot" / "cot" / "self_reflection"

    返回:
        str: 完整的 Prompt 文本
    """
    if task == "merge_prediction":
        return _build_merge_prompt(context, prompt_type)
    elif task == "review_comment":
        return _build_review_prompt(context, prompt_type)
    else:
        raise ValueError(f"Unknown task: {task}")


# ========================================================================
#  Merge Prediction 的 4 种 Prompt
# ========================================================================

def _build_merge_prompt(context, prompt_type):
    if prompt_type == "role_based":
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
            "Analyze the following code change step by step, then determine whether the "
            "Pull Request is likely to be merged.\n\n"
            "Follow these steps:\n"
            "1. Purpose: What is the goal of this code change?\n"
            "2. Correctness: Is the logic correct? Are there potential bugs?\n"
            "3. Security: Are there security risks (injection, permission bypass, etc.)?\n"
            "4. Code Style: Does the code follow project conventions?\n"
            "5. Conclusion: Based on the above, should this PR be merged?\n\n"
            f"Code Change:\n{context}\n\n"
            "Provide your step-by-step analysis first, then output the FINAL JSON on a "
            "separate line. Do NOT wrap the JSON in markdown code blocks (no ```json).\n\n"
            "FINAL JSON:\n"
            '{"decision": "Yes" or "No", "reason": "summary reason"}'
        )

    elif prompt_type == "self_reflection":
        return (
            "You are a code reviewer. For the following code change, perform a TWO-STAGE review:\n\n"
            "STAGE 1 — INITIAL JUDGMENT:\n"
            "Make your initial assessment about whether this PR should be merged. "
            "Consider correctness, security, code style, and maintainability.\n\n"
            "STAGE 2 — SELF-REFLECTION:\n"
            "Critically examine your initial judgment. Ask yourself:\n"
            "- Did I miss any edge cases or potential bugs?\n"
            "- Did I overlook security concerns?\n"
            "- Did I consider code style and maintainability?\n"
            "- Is my reasoning consistent with the evidence?\n"
            "If you find any issues, correct them in your final judgment.\n\n"
            "IMPORTANT: Output EXACTLY in this format, with each section clearly marked:\n\n"
            "[INITIAL JUDGMENT]\n"
            '{"decision": "Yes" or "No", "reason": "brief reason"}\n\n'
            "[REFLECTION]\n"
            "Your critical self-review of the initial judgment...\n\n"
            "[FINAL JUDGMENT]\n"
            '{"decision": "Yes" or "No", "reason": "final reason"}\n\n'
            "Code Change:\n"
            f"{context}"
        )

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


# ========================================================================
#  Review Comment 的 4 种 Prompt
# ========================================================================

def _build_review_prompt(context, prompt_type):
    if prompt_type == "role_based":
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
            "Provide your step-by-step analysis first, then write the FINAL REVIEW COMMENT "
            "on a separate line. Do NOT wrap it in markdown code blocks.\n\n"
            "FINAL REVIEW COMMENT:"
        )

    elif prompt_type == "self_reflection":
        return (
            "You are a code reviewer. For the following code change, perform a TWO-STAGE review:\n\n"
            "STAGE 1 — INITIAL REVIEW:\n"
            "Write your initial review comment for this code change. "
            "Consider correctness, security, style, readability, and maintainability.\n\n"
            "STAGE 2 — SELF-REFLECTION:\n"
            "Critically examine your review comment. Ask yourself:\n"
            "- Did I provide specific, actionable feedback?\n"
            "- Did I miss any important issues?\n"
            "- Is my tone constructive and professional?\n"
            "- Did I suggest concrete improvements?\n"
            "If you find issues, improve your review comment.\n\n"
            "IMPORTANT: Output EXACTLY in this format:\n\n"
            "[INITIAL REVIEW]\n"
            "Your initial review comment...\n\n"
            "[REFLECTION]\n"
            "Your critical self-review of the initial comment...\n\n"
            "[FINAL REVIEW]\n"
            "Your improved, final review comment...\n\n"
            "Code Change:\n"
            f"{context}"
        )

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


# ========================================================================
#  辅助函数
# ========================================================================

def build_all_prompts(context, task):
    """为给定上下文和任务生成全部 4 种 Prompt"""
    return {pt: build_prompt(context, task, pt) for pt in PROMPT_TYPES}


def print_prompt_examples(contexts, context_type="diff_pr_desc"):
    """打印 Prompt 样例，供 notebook 展示"""
    if not contexts:
        print("No data")
        return

    c = contexts[0]
    ctx = c["contexts"].get(context_type, "")
    if not ctx:
        print(f"Context type '{context_type}' not found, using first available")
        ctx = list(c["contexts"].values())[0]

    print("=" * 60)
    print(f"Prompt Examples (Merge Prediction, {context_type})")
    print("=" * 60)

    for pt in PROMPT_TYPES:
        prompt = build_prompt(ctx, "merge_prediction", pt)
        preview = prompt[:400].replace("\n", "\\n")
        print(f"\n--- {pt} ---")
        print(f"  Length: {len(prompt):,} chars")
        print(f"  Preview: {preview}...")

    print()
    print("=" * 60)
    print(f"Prompt Examples (Review Comment, {context_type})")
    print("=" * 60)

    for pt in PROMPT_TYPES:
        prompt = build_prompt(ctx, "review_comment", pt)
        preview = prompt[:400].replace("\n", "\\n")
        print(f"\n--- {pt} ---")
        print(f"  Length: {len(prompt):,} chars")
        print(f"  Preview: {preview}...")


def print_full_prompt_example(contexts, task="merge_prediction", prompt_type="self_reflection", context_type="diff_pr_desc"):
    """打印一个完整的 Prompt 示例（不截断）"""
    if not contexts:
        print("No data")
        return

    c = contexts[0]
    ctx = c["contexts"].get(context_type, "")
    if not ctx:
        ctx = list(c["contexts"].values())[0]

    prompt = build_prompt(ctx, task, prompt_type)
    print("=" * 60)
    print(f"Full Prompt: task={task}, type={prompt_type}, context={context_type}")
    print(f"Length: {len(prompt):,} chars")
    print("=" * 60)
    print(prompt[:2000])
    if len(prompt) > 2000:
        print(f"\n... (truncated, total {len(prompt):,} chars)")