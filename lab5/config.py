"""
实验五 配置文件
路径配置、数据筛选参数等
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ========== 基础路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAB1_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab1")
LAB2_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab2")
LAB3_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab3")
LAB4_BASE_DIR = os.path.join(os.path.dirname(BASE_DIR), "lab4")

# ========== Lab1 数据路径 ==========
LAB1_DATASET_PATH = os.path.join(LAB1_BASE_DIR, "data", "processed", "dataset.csv")
LAB1_RAW_DIR = os.path.join(LAB1_BASE_DIR, "data", "raw")

# ========== Lab2 数据路径 ==========
LAB2_MODELS_DIR = os.path.join(LAB2_BASE_DIR, "data", "models")
LAB2_SVM_MODEL_PATH = os.path.join(LAB2_MODELS_DIR, "svm_model.joblib")
LAB2_RF_MODEL_PATH = os.path.join(LAB2_MODELS_DIR, "rf_model.joblib")
LAB2_SCALER_PATH = os.path.join(LAB2_MODELS_DIR, "scaler.joblib")
LAB2_FEATURES_PATH = os.path.join(LAB2_BASE_DIR, "data", "features", "human_features.csv")
LAB2_CODE_DIR = os.path.join(LAB2_BASE_DIR, "data", "code", "human")
LAB2_AST_DIR = os.path.join(LAB2_BASE_DIR, "data", "ast", "human")
LAB2_CFG_DIR = os.path.join(LAB2_BASE_DIR, "data", "cfg", "human")

# ========== Lab3 数据路径 ==========
LAB3_MODELS_DIR = os.path.join(LAB3_BASE_DIR, "data", "results", "models")
LAB3_CODE2VEC_MODEL_PATH = os.path.join(LAB3_MODELS_DIR, "code2vec_model.pt")
LAB3_CODEBERT_MODEL_DIR = os.path.join(LAB3_BASE_DIR, "data", "codebert_models", "codebert_merge")
LAB3_CODET5_MODEL_DIR = os.path.join(LAB3_BASE_DIR, "data", "codebert_models", "codet5_review")
LAB3_VECTORS_DIR = os.path.join(LAB3_BASE_DIR, "data", "vectors")
LAB3_CODEBERT_DATA_DIR = os.path.join(LAB3_BASE_DIR, "data", "codebert")

# ========== Lab4 数据路径 ==========
LAB4_RESULTS_PATH = os.path.join(LAB4_BASE_DIR, "data", "results.json")
LAB4_CONTEXTS_PATH = os.path.join(LAB4_BASE_DIR, "data", "contexts.json")

# ========== Lab5 输出路径 ==========
LAB5_DATA_DIR = os.path.join(BASE_DIR, "data")
LAB5_FIGURES_DIR = os.path.join(BASE_DIR, "figures")
LAB5_RESULTS_DIR = os.path.join(BASE_DIR, "results")

AI_PRS_PATH = os.path.join(LAB5_DATA_DIR, "ai_prs.csv")
HUMAN_PRS_PATH = os.path.join(LAB5_DATA_DIR, "human_prs.csv")
AI_STATS_PATH = os.path.join(LAB5_DATA_DIR, "ai_stats.json")
COMPARISON_RESULTS_PATH = os.path.join(LAB5_RESULTS_DIR, "comparison_results.json")

os.makedirs(LAB5_DATA_DIR, exist_ok=True)
os.makedirs(LAB5_FIGURES_DIR, exist_ok=True)
os.makedirs(LAB5_RESULTS_DIR, exist_ok=True)

# ========== 目标项目列表（与 lab1 保持一致）==========
TARGET_REPOS = [
    ("kubernetes", "kubernetes"),
    ("pytorch", "pytorch"),
    ("tensorflow", "tensorflow"),
    ("microsoft", "vscode"),
    ("langchain-ai", "langchain"),
]

# ========== 随机种子 ==========
RANDOM_SEED = 42

# ========== Lab5 步骤二：数据准备输出路径 ==========
AI_PULLS_DIR = os.path.join(LAB5_DATA_DIR, "ai_pulls")
AI_CODE_DIR = os.path.join(LAB5_DATA_DIR, "code", "ai")
AI_AST_DIR = os.path.join(LAB5_DATA_DIR, "ast", "ai")
AI_CFG_DIR = os.path.join(LAB5_DATA_DIR, "cfg", "ai")
AI_FEATURES_PATH = os.path.join(LAB5_DATA_DIR, "features", "ai_features.csv")
LAB2_FEATURE_COLUMNS_PATH = os.path.join(LAB2_BASE_DIR, "data", "features", "feature_columns.txt")

# ========== 步骤二结果输出 ==========
STEP2_RESULTS_PATH = os.path.join(LAB5_RESULTS_DIR, "step2_ml_results.json")

# ========== 目录创建 ==========
for dir_path in [AI_PULLS_DIR, os.path.join(LAB5_DATA_DIR, "code", "ai"),
                os.path.join(LAB5_DATA_DIR, "ast", "ai"),
                os.path.join(LAB5_DATA_DIR, "cfg", "ai"),
                os.path.join(LAB5_DATA_DIR, "features")]:
    os.makedirs(dir_path, exist_ok=True)

# ========== 从 lab2 导入配置 ==========
SUPPORTED_EXTENSIONS = {
    ".go": "Go",
    ".py": "Python", ".pyi": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++",
    ".hpp": "C++",
}

# 常见控制流节点类型（覆盖大多数语言）
CONTROL_FLOW_TYPES = {
    "if_statement", "for_statement", "while_statement", "do_statement",
    "switch_statement", "case_statement", "return_statement",
    "break_statement", "continue_statement", "goto_statement",
    "try_statement", "catch_clause", "finally_clause", "with_statement"
}

# tree-sitter 语言名称映射
TREE_SITTER_LANG_MAP = {
    "Go": "go",
    "Python": "python",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "Java": "java",
    "C": "c",
    "C++": "cpp",
}

# GitHub API 配置
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REQUEST_DELAY = 0.5

# ========== Ollama 本地模型配置 ==========
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:7b"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_MAX_TOKENS = 256
MAX_RETRIES = 3
REQUEST_INTERVAL = 0.1
REQUEST_TIMEOUT = 300
MAX_PROMPT_CHARS = 6000

# 代理配置（与 lab2 一致）
HTTP_PROXY = None
HTTPS_PROXY = None

# ========== Lab3 参数配置（与 lab3 一致） ==========
# Code2Vec
MAX_PATH_LENGTH = 8
MAX_PATHS_PER_FILE = 1000
MAX_LEAVES_PER_FILE = 500
CODE_VECTOR_DIM = 128
TOKEN_EMBED_DIM = 64
PATH_EMBED_DIM = 64
CONTEXT_DIM = 128
MIN_PATH_FREQ = 5

# CodeBERT
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODEBERT_MAX_LENGTH = 512

# CodeT5
CODET5_MODEL_NAME = "Salesforce/codet5-base"
CODET5_MAX_INPUT_LENGTH = 1024
CODET5_MAX_TARGET_LENGTH = 256

# ========== Lab3 输出路径（lab5 结果） ==========
STEP3_TASK1_CODE2VEC_PATH = os.path.join(LAB5_RESULTS_DIR, "step3_task1_code2vec.json")
STEP3_TASK2_CODEBERT_PATH = os.path.join(LAB5_RESULTS_DIR, "step3_task2_codebert.json")
STEP3_TASK3_CODET5_PATH = os.path.join(LAB5_RESULTS_DIR, "step3_task3_codet5.json")
STEP3_CACHE_VECTORS = os.path.join(LAB5_DATA_DIR, "step3_vectors.pt")

# AST 路径缓存目录（lab3 模块使用 config.AST_CACHE_DIR）
AST_CACHE_DIR = os.path.join(LAB3_VECTORS_DIR, "ast_cache")
STEP3_AST_CACHE_DIR = AST_CACHE_DIR

# ========== lab3 模块所需路径（从 lab3 导入） ==========
VOCAB_PATH = os.path.join(LAB3_VECTORS_DIR, "vocab.json")
VECTORS_DIR = LAB3_VECTORS_DIR
TRAIN_VECTORS_PATH = os.path.join(LAB3_VECTORS_DIR, "train_vectors.pt")
VAL_VECTORS_PATH = os.path.join(LAB3_VECTORS_DIR, "val_vectors.pt")
TEST_VECTORS_PATH = os.path.join(LAB3_VECTORS_DIR, "test_vectors.pt")
TRAIN_JSON_PATH = os.path.join(LAB3_BASE_DIR, "data", "train.json")
VAL_JSON_PATH = os.path.join(LAB3_BASE_DIR, "data", "val.json")
TEST_JSON_PATH = os.path.join(LAB3_BASE_DIR, "data", "test.json")
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
LAB2_HUMAN_RAW_DIR = os.path.join(LAB2_BASE_DIR, "data", "raw", "human")

# lab3 分类器参数
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_CLASS_WEIGHT = "balanced"
MLP_HIDDEN_LAYERS = (128, 64)
MLP_MAX_ITER = 500
MLP_LEARNING_RATE = 0.001
MLP_BATCH_SIZE = 32
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.1
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# CodeBERT/CodeT5 训练参数
CODEBERT_BATCH_SIZE = 8
CODET5_BATCH_SIZE = 4
CODET5_LEARNING_RATE = 5e-5
CODET5_EPOCHS = 5
CODET5_WEIGHT_DECAY = 0.01
CODET5_WARMUP_STEPS = 100
CODET5_GRADIENT_ACCUMULATION_STEPS = 2
CODET5_MODEL_DIR = LAB3_CODET5_MODEL_DIR

# 输出目录（lab3 模块使用）
RESULTS_DIR = os.path.join(LAB3_BASE_DIR, "data", "results")
PLOTS_DIR = os.path.join(LAB3_BASE_DIR, "image")
MODELS_DIR = LAB3_MODELS_DIR