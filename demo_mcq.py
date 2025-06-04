import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scorer import HalluNoVoScorer

model_name = "meta-llama/Llama-7b-hf"
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
scorer = HalluNoVoScorer(model, tok)

def mcq_score(prompt, options):
    prefix_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    scores = []
    for opt in options:
        ids = torch.cat([prefix_ids,
                         tok(opt, return_tensors="pt").input_ids.cuda()], dim=-1)
        score_tok, conf = scorer.score_next(ids)
        scores.append((opt, conf))
    return sorted(scores, key=lambda x: -x[1])

if __name__ == "__main__":
    q = "Which city is the capital of Australia?\n"
    opts = ["Sydney", "Canberra", "Melbourne", "Perth"]
    print(mcq_score(q, opts))
