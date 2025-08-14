import os, json, random
from typing import List, Dict, Any

class LLMProposer:
    """
    Proposes new candidate formulas from:
    - local heuristics (works offline)
    - optional remote provider (env LLM_PROVIDER='openai' + OPENAI_API_KEY)
    Output: list of formulas in project DSL ([cond OR/AND cond]).
    """
    def __init__(self, state):
        self.state = state
        self.provider = os.environ.get("LLM_PROVIDER", "local").lower()

    def _local_suggestions(self, base_features: List[str], k: int = 3) -> List[List[Any]]:
        # Mine top genes and compose simple boolean formulas
        top = self.state.gene_top(min_count=5, top_n=10)
        pool_feats = [g.split("_")[0] for g,_ in top if "_" in g] or base_features
        def cond():
            f = random.choice(pool_feats)
            op = random.choice([">","<"])
            thr = random.uniform(-1.5, 1.5)
            return [f, op, thr]
        out = []
        for _ in range(k):
            out.append([cond(), random.choice(["AND","OR"]), cond()])
        return out

    def _remote_suggestions(self, base_features: List[str], k: int = 3) -> List[List[Any]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            return self._local_suggestions(base_features, k)
        try:
            import requests
            sys_prompt = "You generate boolean trading rules using the given feature names."
            user = {
              "features": base_features,
              "k": k,
              "format": "Return a JSON list of formulas where each formula is either [cond, 'AND'|'OR', cond] or [feature, '>'|'<', threshold]."
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model":"gpt-4o-mini","messages":[{"role":"system","content":sys_prompt},{"role":"user","content":json.dumps(user)}]}
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            j = r.json()
            txt = j["choices"][0]["message"]["content"]
            suggestions = json.loads(txt)
            cleaned = []
            for f in suggestions:
                if isinstance(f, list) and (isinstance(f[0], list) or (isinstance(f[0], str) and f[1] in (">","<"))):
                    cleaned.append(f)
            return cleaned[:k] if cleaned else self._local_suggestions(base_features, k)
        except Exception:
            return self._local_suggestions(base_features, k)

    def propose(self, base_features: List[str], k: int = 3) -> List[List[Any]]:
        return self._local_suggestions(base_features, k) if self.provider == "local" else self._remote_suggestions(base_features, k)
