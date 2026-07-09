"""
Code2Vec 模型定义
PyTorch 实现的 Code2Vec 神经网络，将 AST 路径编码为代码向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "..")
import config


class Code2Vec(nn.Module):
    """Code2Vec 模型，将多个 AST 路径上下文编码为单个代码向量"""

    def __init__(self, num_tokens: int, num_paths: int):
        """
        Args:
            num_tokens: token 词汇表大小
            num_paths: 路径词汇表大小
        """
        super().__init__()
        self.token_embed = nn.Embedding(
            num_tokens, config.TOKEN_EMBED_DIM, padding_idx=0
        )
        self.path_embed = nn.Embedding(
            num_paths, config.PATH_EMBED_DIM, padding_idx=0
        )

        input_dim = config.TOKEN_EMBED_DIM * 2 + config.PATH_EMBED_DIM
        self.fc = nn.Linear(input_dim, config.CONTEXT_DIM)
        self.attn = nn.Linear(config.CONTEXT_DIM, 1)
        self.dropout = nn.Dropout(0.25)

        self.token_embed_dim = config.TOKEN_EMBED_DIM
        self.path_embed_dim = config.PATH_EMBED_DIM
        self.context_dim = config.CONTEXT_DIM
        self.output_dim = config.CODE_VECTOR_DIM

    def forward(self, starts, paths, ends):
        """
        前向传播

        Args:
            starts: [batch_size, max_paths] 起始 token ID
            paths:  [batch_size, max_paths] 路径 ID
            ends:   [batch_size, max_paths] 结束 token ID

        Returns:
            code_vector: [batch_size, code_vector_dim] 代码向量
        """
        s_emb = self.token_embed(starts)
        p_emb = self.path_embed(paths)
        e_emb = self.token_embed(ends)

        context = torch.cat([s_emb, p_emb, e_emb], dim=-1)
        context = torch.tanh(self.fc(context))
        context = self.dropout(context)

        attn_scores = self.attn(context).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)

        code_vector = torch.sum(attn_weights.unsqueeze(-1) * context, dim=1)

        return code_vector


def create_model(vocab):
    """根据词汇表创建 Code2Vec 模型"""
    num_tokens = len(vocab.token_to_id)
    num_paths = len(vocab.path_to_id)
    model = Code2Vec(num_tokens, num_paths)
    return model