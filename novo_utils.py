# -*- coding: utf-8 -*-
"""
Head-norm voting 置信度 (NoVo)
"""
import torch

def head_norm_score(att_map, heads_keep=None):
    """
    att_map : (layers, heads, L, L)  - 来自 model(..., output_attentions=True)
    返回     : (V,) pseudo-prob 向量, 取倒数第二层 Attention head-norm
    """
    # 默认取倒数第 2 层
    layer_att = att_map[-2]                      # (H,L,L)
    if heads_keep is not None:
        layer_att = layer_att[heads_keep]        # 只保留挑选好的头
    # 每个 head 的 Query=L-1 行 → L 列
    q = layer_att[:, -1, :]                      # (H,L)
    norm = q.norm(p=2, dim=-1)                   # (H,)
    norm = norm / norm.sum()                     # 归一化作为权重
    # 将权重散到整个词表：这里简化假设越大越可靠；真实实现需映射到 logits
    V = att_map[0].shape[-1]
    score = torch.zeros(V, device=att_map[0].device)
    score[:norm.shape[0]] = norm                 # demo: 填到前 H 个位置
    return score                                 # (V,)
