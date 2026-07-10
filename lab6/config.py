"""
实验六 配置文件
路径配置、数据采样参数、上下文类型等
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ========== 基础路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB1_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab1")
LAB4_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab4")
LAB5_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab5")

# ========== Lab5 数据路径（复用 lab5 的 AI PR 数据）==========
LAB5_AI_PULLS_DIR = os.path.join(LAB5_BASE_DIR, "data", "ai_pulls")

# ========== Lab6 输出路径 ==========
LAB6_DATA_DIR = os.path.join(BASE_DIR, "data")
LAB6_RESULTS_DIR = os.path.join(BASE_DIR, "results")
LAB6_FIGURES_DIR = os.path.join(BASE_DIR, "figures")

SELECTED_PRS_PATH = os.path.join(LAB6_DATA_DIR, "selected_prs.json")
CONTEXTS_PATH = os.path.join(LAB6_DATA_DIR, "contexts.json")

os.makedirs(LAB6_DATA_DIR, exist_ok=True)
os.makedirs(LAB6_RESULTS_DIR, exist_ok=True)
os.makedirs(LAB6_FIGURES_DIR, exist_ok=True)

# ========== 目标项目列表（与 lab1/lab5 保持一致）==========
TARGET_REPOS = [
    ("kubernetes", "kubernetes"),
    ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"),
    ("microsoft", "vscode"),
    ("langchain-ai", "langchain"),
]

# ========== 采样参数 ==========
MAX_PRS = 100
RANDOM_SEED = 42

# ========== 上下文类型（lab6 扩展为 5 种）==========
CONTEXT_TYPES = [
    "diff_only",
    "diff_pr_desc",
    "diff_repo",
    "diff_issue",
    "diff_full",
]

# ========== Prompt 类型（lab6 替换 zero_shot 为 self_reflection）==========
PROMPT_TYPES = [
    "few_shot",
    "cot",
    "role_based",
    "self_reflection",
]

# ========== 任务类型 ==========
TASK_TYPES = ["merge_prediction", "review_comment"]

# ========== LLM 推理参数 ==========
LLM_MODEL = "qwen2.5-coder:7b"
LLM_API_URL = "http://localhost:11434/api/generate"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512
LLM_MAX_PROMPT_CHARS = 25000
LLM_MAX_RETRIES = 3
LLM_REQUEST_TIMEOUT = 300
LLM_REQUEST_INTERVAL = 0.1

# ========== Issue 引用匹配模式 ==========
ISSUE_REFERENCE_PATTERNS = [
    r'(?:fix|fixes|fixed|close|closes|closed|resolve|resolves|resolved|ref|reference|issue|see|addresses|address|related)\s+#(\d+)',
    r'#(\d+)',
]

# ========== 语言检测映射 ==========
LANGUAGE_MAP = {
    ".py": "Python",
    ".pyi": "Python",
    ".go": "Go",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".rs": "Rust",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".bzl": "Starlark",
    ".BUILD": "Starlark",
    ".proto": "Protobuf",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".md": "Markdown",
    ".txt": "Text",
    ".sh": "Shell",
    ".cmake": "CMake",
    ".dockerfile": "Dockerfile",
}

# ========== 过滤关键词（排除非函数名）==========
FILTER_KEYWORDS = {
    "if", "for", "while", "switch", "return", "import", "from",
    "print", "assert", "with", "def", "class", "function", "func",
    "try", "except", "catch", "finally", "raise", "yield", "await",
    "async", "new", "delete", "this", "super", "self", "true",
    "false", "null", "none", "else", "elif", "case", "break",
    "continue", "pass", "throw", "static", "public", "private",
    "protected", "void", "int", "float", "double", "bool", "char",
    "string", "var", "let", "const", "len", "range", "type",
    "sizeof", "typeof", "instanceof",
}