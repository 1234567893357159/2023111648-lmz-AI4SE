"""
AST 路径提取模块
使用 tree-sitter 解析代码，提取 Code2Vec 所需的 AST 路径上下文
"""

import os
import random

from tree_sitter import Parser
import tree_sitter_languages

import sys
sys.path.insert(0, "..")
import config


class ASTPathExtractor:
    """AST 路径提取器，从代码文件中提取 AST 路径上下文"""

    def __init__(self):
        self._parser_cache = {}

    def _detect_language(self, filepath: str):
        """根据文件扩展名检测语言"""
        ext = os.path.splitext(filepath)[1].lower()
        return config.SUPPORTED_EXTENSIONS.get(ext)

    def _get_parser(self, lang_name: str):
        """获取 tree-sitter 解析器（带缓存）"""
        ts_lang = config.TREE_SITTER_LANG_MAP.get(lang_name)
        if ts_lang is None:
            return None
        if ts_lang not in self._parser_cache:
            parser = Parser()
            parser.set_language(tree_sitter_languages.get_language(ts_lang))
            self._parser_cache[ts_lang] = parser
        return self._parser_cache[ts_lang]

    def _is_leaf_node(self, node) -> bool:
        """判断是否为叶子节点（终端 token）"""
        if node.child_count == 0:
            return True
        for child in node.children:
            if child.is_named:
                return False
        return True

    def _get_token(self, node, code_bytes: bytes) -> str:
        """获取节点的 token 文本"""
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _collect_leaves(self, node, code_bytes: bytes, leaves: list, parent_map: dict):
        """迭代收集 AST 中的所有叶子节点（避免递归深度过大）"""
        max_leaves = config.MAX_LEAVES_PER_FILE * 2
        stack = [(node, None)]
        while stack and len(leaves) < max_leaves:
            current, parent = stack.pop()
            if parent is not None:
                parent_map[current] = parent
            if current.is_named:
                children = list(current.children)
                for child in reversed(children):
                    stack.append((child, current))
            if self._is_leaf_node(current):
                token = self._get_token(current, code_bytes)
                if token.strip():
                    leaves.append(current)

    def _get_ancestor_path(self, node, parent_map: dict):
        """获取从节点到根节点的祖先路径"""
        path = []
        current = node
        while current in parent_map:
            parent = parent_map[current]
            path.append(parent.type)
            current = parent
        return path

    def _build_path(self, leaf_a, leaf_b, parent_map: dict):
        """构建两个叶子节点之间的 AST 路径"""
        ancestors_a = self._get_ancestor_path(leaf_a, parent_map)
        ancestors_b = self._get_ancestor_path(leaf_b, parent_map)

        i = 1
        while i <= min(len(ancestors_a), len(ancestors_b)):
            if ancestors_a[-i] != ancestors_b[-i]:
                break
            i += 1
        lca_depth = i - 1

        up_part = ancestors_a[:-lca_depth] if lca_depth > 0 else ancestors_a
        down_part = ancestors_b[:-lca_depth] if lca_depth > 0 else ancestors_b

        if len(up_part) + len(down_part) > config.MAX_PATH_LENGTH:
            return None

        path_parts = []
        for t in up_part:
            path_parts.append(f"UP_{t}")
        for t in reversed(down_part):
            path_parts.append(f"DOWN_{t}")

        return "|".join(path_parts)

    def _extract_paths_from_tree(self, tree, code_bytes: bytes):
        """从一棵 AST 中提取路径上下文"""
        root = tree.root_node
        leaves = []
        parent_map = {}
        self._collect_leaves(root, code_bytes, leaves, parent_map)

        if len(leaves) < 2:
            return []

        paths = []
        max_pairs = config.MAX_PATHS_PER_FILE * 2
        leaf_count = len(leaves)

        if leaf_count <= 20:
            for i in range(leaf_count):
                for j in range(i + 1, leaf_count):
                    if len(paths) >= config.MAX_PATHS_PER_FILE:
                        return paths
                    start_token = self._get_token(leaves[i], code_bytes)
                    end_token = self._get_token(leaves[j], code_bytes)
                    path_str = self._build_path(leaves[i], leaves[j], parent_map)
                    if path_str and len(path_str.split("|")) <= config.MAX_PATH_LENGTH:
                        paths.append((start_token, path_str, end_token))
        else:
            for _ in range(max_pairs):
                if len(paths) >= config.MAX_PATHS_PER_FILE:
                    break
                i, j = random.sample(range(leaf_count), 2)
                if i > j:
                    i, j = j, i
                start_token = self._get_token(leaves[i], code_bytes)
                end_token = self._get_token(leaves[j], code_bytes)
                path_str = self._build_path(leaves[i], leaves[j], parent_map)
                if path_str and len(path_str.split("|")) <= config.MAX_PATH_LENGTH:
                    paths.append((start_token, path_str, end_token))

        return paths

    def extract_paths(self, filepath: str):
        """从单个文件中提取 AST 路径上下文"""
        lang = self._detect_language(filepath)
        if lang is None:
            return []

        parser = self._get_parser(lang)
        if parser is None:
            return []

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception:
            return []

        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)

        return self._extract_paths_from_tree(tree, code_bytes)

    def extract_paths_from_dir(self, dir_path: str):
        """从目录中所有代码文件提取 AST 路径上下文"""
        all_paths = []
        if not os.path.exists(dir_path):
            return all_paths

        for fname in os.listdir(dir_path):
            if fname.endswith(".notfound"):
                continue
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            lang = self._detect_language(fpath)
            if lang is None:
                continue
            paths = self.extract_paths(fpath)
            all_paths.extend(paths)

        return all_paths

    # ========== 叶子数据缓存相关 ==========
    # 缓存 AST 叶子节点的原始数据（token + 祖先类型列表），
    # 而不是截断后的路径。这样修改 MAX_PATHS_PER_FILE 后无需重新解析代码。

    def _extract_leaf_data_from_tree(self, tree, code_bytes: bytes):
        """从 AST 树中提取叶子节点数据：[token, [ancestor_type, ...]]"""
        root = tree.root_node
        leaves = []
        parent_map = {}
        self._collect_leaves(root, code_bytes, leaves, parent_map)

        leaf_data = []
        for leaf in leaves:
            token = self._get_token(leaf, code_bytes)
            if token.strip():
                ancestors = self._get_ancestor_path(leaf, parent_map)
                leaf_data.append([token, ancestors])

        max_leaves = config.MAX_LEAVES_PER_FILE
        if len(leaf_data) > max_leaves:
            import random as _random
            leaf_data = _random.sample(leaf_data, max_leaves)

        return leaf_data

    def extract_leaf_data(self, filepath: str):
        """从单个文件中提取叶子节点数据（用于缓存）"""
        lang = self._detect_language(filepath)
        if lang is None:
            return []

        parser = self._get_parser(lang)
        if parser is None:
            return []

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception:
            return []

        code_bytes = code.encode("utf-8")
        tree = parser.parse(code_bytes)

        return self._extract_leaf_data_from_tree(tree, code_bytes)

    def extract_leaf_data_from_dir(self, dir_path: str):
        """从目录中所有代码文件提取叶子节点数据"""
        all_leaf_data = []
        if not os.path.exists(dir_path):
            return all_leaf_data

        for fname in os.listdir(dir_path):
            if fname.endswith(".notfound"):
                continue
            fpath = os.path.join(dir_path, fname)
            if not os.path.isfile(fpath):
                continue
            lang = self._detect_language(fpath)
            if lang is None:
                continue
            leaf_data = self.extract_leaf_data(fpath)
            all_leaf_data.extend(leaf_data)

        return all_leaf_data

    def _build_path_from_ancestors(self, ancestors_a: list, ancestors_b: list):
        """从两个叶子节点的祖先列表构建 AST 路径（不依赖 tree-sitter 节点）"""
        i = 1
        while i <= min(len(ancestors_a), len(ancestors_b)):
            if ancestors_a[-i] != ancestors_b[-i]:
                break
            i += 1
        lca_depth = i - 1

        up_part = ancestors_a[:-lca_depth] if lca_depth > 0 else ancestors_a
        down_part = ancestors_b[:-lca_depth] if lca_depth > 0 else ancestors_b

        if len(up_part) + len(down_part) > config.MAX_PATH_LENGTH:
            return None

        path_parts = []
        for t in up_part:
            path_parts.append(f"UP_{t}")
        for t in reversed(down_part):
            path_parts.append(f"DOWN_{t}")

        return "|".join(path_parts)

    def extract_paths_from_leaf_data(self, leaf_data: list, max_paths: int = None):
        """从缓存的叶子数据中提取 AST 路径（可按需重新采样）"""
        if max_paths is None:
            max_paths = config.MAX_PATHS_PER_FILE

        if len(leaf_data) < 2:
            return []

        paths = []
        max_pairs = max_paths * 2
        leaf_count = len(leaf_data)

        if leaf_count <= 20:
            for i in range(leaf_count):
                for j in range(i + 1, leaf_count):
                    if len(paths) >= max_paths:
                        return paths
                    start_token, ancestors_a = leaf_data[i]
                    end_token, ancestors_b = leaf_data[j]
                    path_str = self._build_path_from_ancestors(ancestors_a, ancestors_b)
                    if path_str:
                        paths.append((start_token, path_str, end_token))
        else:
            for _ in range(max_pairs):
                if len(paths) >= max_paths:
                    break
                i, j = random.sample(range(leaf_count), 2)
                if i > j:
                    i, j = j, i
                start_token, ancestors_a = leaf_data[i]
                end_token, ancestors_b = leaf_data[j]
                path_str = self._build_path_from_ancestors(ancestors_a, ancestors_b)
                if path_str:
                    paths.append((start_token, path_str, end_token))

        return paths