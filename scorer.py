# -*- coding: utf-8 -*-
import torch, torch.nn.functional as F
from hallu_core import HalluCanary
from novo_utils import head_norm_score

class HalluNoVoScorer:
    def __init__(self, model, tokenizer,
                 w_hallu: float = 0.6,
                 w_novo:  float = 0.4):
        self.model, self.tok = model, tokenizer
        self.canary = HalluCanary(model, tokenizer)
        self.w_hallu = w_hallu
        self.w_novo  = w_novo

    @torch.no_grad()
    def score_next(self, prefix_ids: torch.Tensor):
        """
        返回：(next_token_id, fused_score)
        """
        # ① NoVo 置信度向量
        outs = self.model(prefix_ids, output_attentions=True)
        score_novo = torch.softmax(
            head_norm_score(outs.attentions), dim=-1)              # (V,)

        # ② HALLUCANA 多分支风险
        cand_ids, risk_vec = self.canary.step_risk(prefix_ids)     # (k,3)
        risk_hallu = risk_vec.mean(-1)                             # (k,)  简单平均
        prob_hallu = torch.exp(-risk_hallu)                        # 变成“置信度”
        score_hallu = torch.zeros_like(score_novo)
        score_hallu[cand_ids] = prob_hallu / prob_hallu.sum()

        # ③ 融合
        fused = self.w_novo * score_novo + self.w_hallu * score_hallu
        tok = fused.argmax()
        return tok.item(), fused[tok].item()
