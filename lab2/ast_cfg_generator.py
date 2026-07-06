"""
AST 与 CFG 生成模块
对已下载的代码文件，使用 tree-sitter 解析生成 AST 和 CFG，保存到本地

输出目录结构（与 code/ 同级）：
data/
  code/human/{owner}_{repo}/{pr_id}/before/...   ← 原始代码
  code/human/{owner}_{repo}/{pr_id}/after/...
  ast/human/{owner}_{repo}/{pr_id}/before/...    ← AST JSON
  ast/human/{owner}_{repo}/{pr_id}/after/...
  cfg/human/{owner}_{repo}/{pr_id}/before/...    ← CFG JSON
  cfg/human/{owner}_{repo}/{pr_id}/after/...
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import tree_sitter_languages
from tree_sitter import Parser
from tqdm import tqdm

import config


class ASTCFGGenerator:
    """AST 与 CFG 生成器：解析代码文件，生成 AST 和 CFG 并保存"""

    def __init__(self):
        self.stats = {
            "total": 0,
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "unsupported": 0,
        }
        self._parser_cache = {}

    def _get_parser(self, lang: str) -> Optional[Parser]:
        """获取缓存的语言解析器"""
        ts_lang = config.TREE_SITTER_LANG_MAP.get(lang)
        if ts_lang is None:
            return None
        if ts_lang not in self._parser_cache:
            try:
                parser = Parser()
                tree_sitter_lang = tree_sitter_languages.get_language(ts_lang)
                parser.set_language(tree_sitter_lang)
                self._parser_cache[ts_lang] = parser
            except Exception as e:
                print(f"  ⚠️ 无法加载 {lang} ({ts_lang}) 解析器: {e}")
                return None
        return self._parser_cache[ts_lang]

    def _get_language(self, filepath: str) -> Optional[str]:
        """根据扩展名判断语言"""
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            return None
        return config.SUPPORTED_EXTENSIONS.get(ext)

    def _parse_file(self, filepath: str, lang: str):
        """解析文件，返回 tree-sitter AST 根节点"""
        parser = self._get_parser(lang)
        if parser is None:
            return None
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        return parser.parse(bytes(code, "utf-8"))

    def _count_nodes(self, node) -> int:
        """递归统计 AST 节点数量"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _extract_control_flow_nodes(self, node) -> List[Dict]:
        """递归提取控制流节点信息"""
        results = []
        if node.type in config.CONTROL_FLOW_TYPES:
            results.append({
                "type": node.type,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
                "start_byte": node.start_byte,
                "end_byte": node.end_byte,
            })
        for child in node.children:
            results.extend(self._extract_control_flow_nodes(child))
        return results

    def _ast_to_json(self, root_node, lang: str) -> Dict:
        """将 AST 根节点转为可保存的 JSON 结构"""
        control_flow_nodes = self._extract_control_flow_nodes(root_node)
        node_count = self._count_nodes(root_node)
        return {
            "language": lang,
            "sexp": root_node.sexp(),
            "node_count": node_count,
            "control_flow_nodes": control_flow_nodes,
            "control_flow_count": len(control_flow_nodes),
        }

    def _find_functions(self, node) -> List[Dict]:
        """查找所有函数/方法定义节点"""
        func_types = {
            "function_declaration",
            "function_definition",
            "method_declaration",
            "method_definition",
            "constructor_declaration",
            "arrow_function",
            "function",
            "lambda",
        }
        results = []
        if node.type in func_types:
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode("utf-8") if name_node else "anonymous"
            results.append({
                "name": name,
                "node": node,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1,
            })
        for child in node.children:
            results.extend(self._find_functions(child))
        return results

    def _collect_statements(self, node) -> List:
        """收集函数体内的语句节点（用于构建基本块）"""
        stmt_types = {
            "expression_statement", "declaration", "return_statement",
            "if_statement", "for_statement", "while_statement",
            "do_statement", "switch_statement", "try_statement",
            "break_statement", "continue_statement", "goto_statement",
            "labeled_statement", "throw_statement", "assert_statement",
            "variable_declaration", "assignment_expression",
            "call_expression", "with_statement",
        }
        stmts = []
        for child in node.children:
            if child.type in stmt_types:
                stmts.append(child)
            elif child.type in ("{", "}", "(", ")", ";", ":"):
                continue
            else:
                stmts.extend(self._collect_statements(child))
        return stmts

    def _build_cfg_for_function(self, func_info: Dict) -> Dict:
        """为单个函数构建 CFG"""
        stmts = self._collect_statements(func_info["node"])
        if not stmts:
            return None

        graph = nx.DiGraph()
        node_id = 0

        graph.add_node(node_id, type="entry", label="entry",
                       line=func_info["start_line"])
        entry_id = node_id
        node_id += 1

        prev_ids = [entry_id]

        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            stmt_type = stmt.type
            stmt_line = stmt.start_point[0] + 1

            current_id = node_id
            node_id += 1
            graph.add_node(current_id, type=stmt_type,
                           label=stmt_type, line=stmt_line)

            for pid in prev_ids:
                graph.add_edge(pid, current_id)
            prev_ids = [current_id]

            if stmt_type == "if_statement":
                consequence = stmt.child_by_field_name("consequence")
                alternative = stmt.child_by_field_name("alternative")

                fork_ids = []
                for branch in [consequence, alternative]:
                    if branch:
                        branch_stmts = self._collect_statements(branch)
                        if branch_stmts:
                            branch_first = node_id
                            prev_ids = [current_id]
                            for bs in branch_stmts:
                                bs_id = node_id
                                node_id += 1
                                graph.add_node(bs_id, type=bs.type,
                                               label=bs.type,
                                               line=bs.start_point[0] + 1)
                                for pid in prev_ids:
                                    graph.add_edge(pid, bs_id)
                                prev_ids = [bs_id]
                            fork_ids.append(prev_ids[0])
                        else:
                            fork_ids.append(current_id)
                    else:
                        fork_ids.append(current_id)
                prev_ids = fork_ids

                while i + 1 < len(stmts):
                    next_stmt = stmts[i + 1]
                    if next_stmt.start_point[0] > stmt.end_point[0]:
                        break
                    i += 1

            elif stmt_type in ("for_statement", "while_statement", "do_statement"):
                body = stmt.child_by_field_name("body")
                if body:
                    body_stmts = self._collect_statements(body)
                    if body_stmts:
                        body_first = node_id
                        prev_ids = [current_id]
                        for bs in body_stmts:
                            bs_id = node_id
                            node_id += 1
                            graph.add_node(bs_id, type=bs.type,
                                           label=bs.type,
                                           line=bs.start_point[0] + 1)
                            for pid in prev_ids:
                                graph.add_edge(pid, bs_id)
                            prev_ids = [bs_id]
                        graph.add_edge(prev_ids[0], current_id)
                        prev_ids = [current_id]

                while i + 1 < len(stmts):
                    next_stmt = stmts[i + 1]
                    if next_stmt.start_point[0] > stmt.end_point[0]:
                        break
                    i += 1

            elif stmt_type in ("return_statement", "break_statement",
                               "continue_statement", "goto_statement",
                               "throw_statement"):
                prev_ids = []

            i += 1

        if prev_ids:
            exit_id = node_id
            graph.add_node(exit_id, type="exit", label="exit",
                           line=func_info["end_line"])
            for pid in prev_ids:
                graph.add_edge(pid, exit_id)

        if graph.number_of_nodes() <= 2:
            return None

        return {
            "name": func_info["name"],
            "start_line": func_info["start_line"],
            "end_line": func_info["end_line"],
            "graph": nx.node_link_data(graph),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }

    def _build_cfg(self, root_node, lang: str) -> Dict:
        """从 AST 根节点构建 CFG"""
        functions = self._find_functions(root_node)
        func_cfgs = []
        for func in functions:
            cfg = self._build_cfg_for_function(func)
            if cfg:
                func_cfgs.append(cfg)

        total_nodes = sum(f["node_count"] for f in func_cfgs)
        total_edges = sum(f["edge_count"] for f in func_cfgs)

        return {
            "language": lang,
            "functions": func_cfgs,
            "function_count": len(func_cfgs),
            "total_node_count": total_nodes,
            "total_edge_count": total_edges,
        }

    def _ensure_output_dir(self, code_path: str, output_type: str) -> str:
        """根据代码文件路径，构造对应的 AST/CFG 输出路径
        
        例: data/code/human/kubernetes_kubernetes/139699/before/file.go
          → data/ast/human/kubernetes_kubernetes/139699/before/file.json
        """
        rel = os.path.relpath(code_path, config.HUMAN_CODE_DIR)
        if output_type == "ast":
            base_dir = config.HUMAN_AST_DIR
        else:
            base_dir = config.HUMAN_CFG_DIR

        out_path = os.path.join(base_dir, rel)
        out_path = os.path.splitext(out_path)[0] + ".json"
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        return out_path

    def _process_file(self, code_path: str, relative: str) -> bool:
        """处理单个代码文件：生成 AST 和 CFG 并保存"""
        lang = self._get_language(code_path)
        if lang is None:
            self.stats["unsupported"] += 1
            return False

        ast_path = self._ensure_output_dir(code_path, "ast")
        cfg_path = self._ensure_output_dir(code_path, "cfg")

        ast_exists = os.path.exists(ast_path)
        cfg_exists = os.path.exists(cfg_path)
        if ast_exists and cfg_exists:
            self.stats["skipped"] += 1
            return True

        tree = self._parse_file(code_path, lang)
        if tree is None:
            self.stats["failed"] += 1
            return False

        root = tree.root_node

        if not ast_exists:
            ast_data = self._ast_to_json(root, lang)
            with open(ast_path, "w", encoding="utf-8") as f:
                json.dump(ast_data, f, ensure_ascii=False)

        if not cfg_exists:
            cfg_data = self._build_cfg(root, lang)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, ensure_ascii=False)

        self.stats["success"] += 1
        return True

    def _collect_code_files(self) -> List[Tuple[str, str]]:
        """收集所有需要处理的代码文件，返回 (path, relative_path) 列表"""
        files = []
        if not os.path.exists(config.HUMAN_CODE_DIR):
            return files
        for root, dirs, filenames in os.walk(config.HUMAN_CODE_DIR):
            for fname in filenames:
                if fname.endswith(".notfound"):
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, config.HUMAN_CODE_DIR)
                files.append((fpath, rel))
        return files

    def run_all(self, force_check: bool = False):
        """运行所有代码文件的 AST 和 CFG 生成
        
        Args:
            force_check: False=跳过所有检查直接返回, True=遍历文件生成 AST/CFG
        """
        if not force_check:
            print("=" * 60)
            print("FORCE_CHECK=False, 跳过 AST/CFG 生成")
            print("=" * 60)
            return

        print("=" * 60)
        print("开始生成 AST 和 CFG...")
        print("=" * 60)

        files = self._collect_code_files()
        self.stats["total"] = len(files)
        print(f"  找到 {len(files)} 个代码文件需要处理\n")

        pbar = tqdm(files, desc="  生成 AST/CFG", unit="file",
                    ncols=100, ascii=True)
        for fpath, rel in pbar:
            self._process_file(fpath, rel)
            pbar.set_postfix({
                "成功": self.stats["success"],
                "跳过": self.stats["skipped"],
                "失败": self.stats["failed"],
            })
        pbar.close()

        print(f"\n{'=' * 60}")
        print(f"AST 和 CFG 生成完成!")
        print(f"  总文件数:     {self.stats['total']}")
        print(f"  生成成功:     {self.stats['success']}")
        print(f"  跳过(已存在): {self.stats['skipped']}")
        print(f"  生成失败:     {self.stats['failed']}")
        print(f"  不支持语言:   {self.stats['unsupported']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    generator = ASTCFGGenerator()
    generator.run_all()