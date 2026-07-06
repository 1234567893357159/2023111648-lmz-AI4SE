"""
测试 tree-sitter 对多种语言的解析能力，并验证是否能提取控制流信息（用于构建 CFG）。
支持：Go, C++, C, Java, JavaScript, Python, TypeScript
"""

import tree_sitter_languages
from tree_sitter import Parser

# 示例代码片段（带 if 分支和递归调用，用于测试控制流）
CODE_SAMPLES = {
    "go": """
package main
func fact(n int) int {
    if n <= 1 {
        return 1
    }
    return n * fact(n-1)
}
""",
    "cpp": """
#include <iostream>
int fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n-1);
}
""",
    "c": """
#include <stdio.h>
int fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n-1);
}
""",
    "java": """
public class Fact {
    public int fact(int n) {
        if (n <= 1) return 1;
        return n * fact(n-1);
    }
}
""",
    "javascript": """
function fact(n) {
    if (n <= 1) return 1;
    return n * fact(n-1);
}
""",
    "python": """
def fact(n):
    if n <= 1:
        return 1
    return n * fact(n-1)
""",
    "typescript": """
function fact(n: number): number {
    if (n <= 1) return 1;
    return n * fact(n-1);
}
"""
}

# 语言名称与 tree-sitter-languages 中注册名称的映射
LANG_MAP = {
    "go": "go",
    "cpp": "cpp",
    "c": "c",
    "java": "java",
    "javascript": "javascript",
    "python": "python",
    "typescript": "typescript"
}

# 常见控制流节点类型（覆盖大多数语言）
CONTROL_FLOW_TYPES = {
    "if_statement", "for_statement", "while_statement", "do_statement",
    "switch_statement", "case_statement", "return_statement",
    "break_statement", "continue_statement", "goto_statement",
    "try_statement", "catch_clause", "finally_clause", "with_statement"
}

def analyze_control_flow(node, depth=0, lang=""):
    """
    递归遍历 AST，收集控制流节点并打印信息。
    返回控制流节点列表（每个元素为 (类型, 起始行, 嵌套深度)）
    """
    results = []
    if node.type in CONTROL_FLOW_TYPES:
        start_line = node.start_point[0] + 1  # 转为 1-based
        indent = "  " * depth
        print(f"{indent}├─ {node.type} (行 {start_line})")
        results.append((node.type, start_line, depth))
    # 递归处理子节点
    for child in node.children:
        results.extend(analyze_control_flow(child, depth + 1, lang))
    return results

def main():
    print("=== Tree-sitter 多语言解析 + 控制流提取测试 ===\n")
    
    for lang_name, code in CODE_SAMPLES.items():
        print(f"--- 语言: {lang_name} ---")
        try:
            # 获取解析器
            parser = Parser()
            lang = tree_sitter_languages.get_language(LANG_MAP[lang_name])
            parser.set_language(lang)
            
            # 解析代码
            tree = parser.parse(bytes(code, "utf-8"))
            root = tree.root_node
            
            # 检查语法错误
            has_error = False
            def check_error(node):
                nonlocal has_error
                if node.type == "ERROR":
                    has_error = True
                    return
                for child in node.children:
                    check_error(child)
            check_error(root)
            
            if has_error:
                print("  ⚠️  解析存在语法错误（可能代码片段不完整）")
            else:
                print("  ✅ 解析成功，无语法错误")
            
            # 打印完整 AST（S-expression）
            print("  AST (S-expression):")
            sexp = root.sexp()
            print(f"  {sexp[:500]}{'...' if len(sexp) > 500 else ''}")
            
            # 打印顶层节点数量
            print(f"  顶层节点数: {len(root.children)}")
            
            # --- 控制流分析 ---
            print("  控制流节点 (CFG 基础信息):")
            cf_nodes = analyze_control_flow(root, depth=0, lang=lang_name)
            if not cf_nodes:
                print("    (未找到控制流节点)")
            else:
                print(f"    共找到 {len(cf_nodes)} 个控制流节点")
            print()
            
        except Exception as e:
            print(f"  ❌ 解析失败: {e}\n")

if __name__ == "__main__":
    main()