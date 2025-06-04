# -*- coding: utf-8 -*-
"""
Minimal HALLUCANA implementation: Canary look-ahead + three风险特征
"""
import torch, torch.nn.functional as F
from typing import List, Tuple

class HalluCanary:
    def __init__(self, model, tokenizer,
                 look_k: int = 4,
                 top_p: float = 0.95):
        self.model, self.tok = model, tokenizer
        self.look_k = look_k
        self.top_p = top_p

    # ====== public ======
    @torch.no_grad()
    def step_risk(self,
                  prefix_ids: torch.Tensor,   # (1, L)
                  temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            cand_ids : (k,)    look-ahead 生成的 token id
            risk_vec : (k, F)  每个候选的多维风险特征
        """
        # (A) 1-step nucleus 采样 k 条
        logits = self.model(prefix_ids).logits[:, -1, :]          # (1,V)
        logits = logits / temp
        probs  = torch.softmax(logits, dim=-1)
        cand_ids = self._top_p_sample(probs, k=self.look_k)       # (k,)
        # (B) 风险 1：-log p  (熟悉度低 → 高风险)
        logp  = probs.log()[0, cand_ids]
        risk1 = -logp                                             # (k,)
        # (C) 风险 2：logit variance （公式：σ²(logits)）
        risk2 = logits.var(dim=-1)[0].expand_as(risk1)
        # (D) 风险 3：句法跳跃（粗略用 token 距 prompt 长度代替）
        pos   = prefix_ids.shape[-1]
        risk3 = (1 / (pos + 1.)) * torch.ones_like(risk1)         # 与位置反比

        risk_vec = torch.stack([risk1, risk2, risk3], dim=-1)     # (k,3)
        return cand_ids, risk_vec

    # ====== helpers ======
    def _top_p_sample(self, probs, k):
        sorted_p, sorted_id = probs.sort(dim=-1, descending=True)
        cumsum = sorted_p.cumsum(-1)
        mask   = cumsum <= self.top_p
        mask[..., :1] = True          # 至少保留一个
        cand_pool = sorted_id[mask]
        # 取 min(len(pool), k)
        pick = cand_pool[torch.randperm(len(cand_pool))[:k]]
        return pick
