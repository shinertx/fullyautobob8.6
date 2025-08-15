# OpenAI-only proposer for boolean trading formulas.
# Prefers Responses API (GPT-5+) and falls back to Chat Completions automatically.

from __future__ import annotations
import os, json, time
from typing import List, Any
from loguru import logger

try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Missing OpenAI SDK. Ensure 'openai>=1.35.0,<2' is in requirements.txt") from e


class LLMProposer:
    """
    Generates candidate boolean trading rules using ONLY OpenAI models.
    Output schema: each formula is either:
      [feature, '>'|'<', float_threshold]  OR
      [ <formula>, 'AND'|'OR', <formula> ]
    Thresholds are floats in [-2.0, 2.0] (post-normalized feature space).
    """

    def __init__(self, state=None):
        self.state = state
        # required key
        self.api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for LLMProposer.")

        # config
        self.model = (os.environ.get("OPENAI_MODEL") or "gpt-5").strip()
        self.base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        try:
            self.temperature = float(os.environ.get("OPENAI_TEMPERATURE") or 0.2)
        except Exception:
            self.temperature = 0.2
        self.use_responses_api = (os.environ.get("OPENAI_USE_RESPONSES", "1").strip() == "1")
        self.max_retries = 3

        # client
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"[LLM] OpenAI enabled | model={self.model} | responses_api={self.use_responses_api}")

    # ---------------------------- public API ---------------------------- #

    def propose(self, base_features: List[str], k: int = 3) -> List[List[Any]]:
        """Return a list of boolean formulas. Raises on failure (no local fallback)."""
        if not base_features:
            raise ValueError("base_features must be a non-empty list of feature names.")
        k = max(1, int(k))

        prompt_user = {
            "features": base_features,
            "k": k,
            "format": "Return a JSON array of length k. Each item is either "
                      "[feature, '>'|' <', threshold_float] or "
                      "[left_formula, 'AND'|'OR', right_formula]. "
                      "feature ∈ features; threshold ∈ [-2.0, 2.0]. No prose—JSON ONLY."
        }

        txt = None
        if self.use_responses_api:
            txt = self._call_responses_api(prompt_user)  # try GPT-5+ path first
            if txt is None:
                logger.warning("[LLM] Responses API failed; falling back to Chat Completions.")
        if txt is None:
            txt = self._call_chat_completions(prompt_user)  # generic chat path

        formulas = self._parse_json_formulas(txt)
        self._validate_formulas(formulas, base_features, k)
        return formulas[:k]

    # ------------------------- OpenAI integrations ---------------------- #

    def _call_responses_api(self, payload: dict) -> str | None:
        """Use Responses API (recommended for GPT-5)."""
        params = {
            "model": self.model,
            "input": json.dumps(payload),
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(**params)
                return getattr(resp, "output_text", None) or self._coerce_text(resp)
            except Exception as e:
                logger.warning(f"[LLM] responses.create attempt {attempt} failed: {e}")
                time.sleep(min(2**attempt, 6))
        return None

    def _call_chat_completions(self, payload: dict) -> str:
        """Use Chat Completions (works broadly across GA chat models)."""
        last_err: Exception | None = None
        params = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You generate boolean trading rules using ONLY the provided feature names."},
                {"role": "user", "content": json.dumps(payload)}
            ],
        }
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**params)
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
                logger.warning(f"[LLM] chat.completions.create attempt {attempt} failed: {e}")
                time.sleep(min(2**attempt, 6))
        raise RuntimeError(f"OpenAI Chat Completions failed after retries: {last_err}")

    # ----------------------------- helpers ------------------------------ #

    @staticmethod
    def _coerce_text(resp_obj) -> str | None:
        # Try to pull a plain string from a Responses API object if output_text is absent.
        try:
            if hasattr(resp_obj, "output") and resp_obj.output and hasattr(resp_obj.output[0], "content"):
                chunks = []
                for part in resp_obj.output[0].content:
                    if hasattr(part, "text") and hasattr(part.text, "value"):
                        chunks.append(part.text.value)
                return "\n".join(chunks) if chunks else None
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_json_formulas(text: str) -> List[List[Any]]:
        if not text or not isinstance(text, str):
            raise ValueError("LLM returned empty response.")
        # Strip any surrounding markdown fences the model might add.
        s = text.strip()
        if s.startswith("```"):
            s = s.strip("`")
            # Remove possible language hint
            first_nl = s.find("\n")
            if first_nl != -1:
                s = s[first_nl+1:]
        try:
            data = json.loads(s)
        except Exception as e:
            # try to salvage a JSON array substring
            start = s.find("[")
            end = s.rfind("]")
            if start != -1 and end != -1:
                data = json.loads(s[start:end+1])
            else:
                raise ValueError(f"Failed to parse LLM JSON: {e}") from e
        if not isinstance(data, list):
            raise ValueError("LLM output is not a JSON array.")
        return data

    @staticmethod
    def _validate_formulas(formulas: List[List[Any]], base_features: List[str], k: int) -> None:
        if len(formulas) < 1:
            raise ValueError("LLM returned no formulas.")
        def _check(node):
            if isinstance(node, list) and len(node) == 3:
                # leaf condition or binary boolean
                if isinstance(node[0], list) and isinstance(node[2], list):
                    # boolean node
                    if node[1] not in ("AND", "OR"):
                        raise ValueError("Invalid boolean operator in formula.")
                    _check(node[0]); _check(node[2])
                else:
                    # leaf
                    feat, op, thr = node
                    if feat not in base_features:
                        raise ValueError(f"Unknown feature in formula: {feat}")
                    if op not in (">", "<"):
                        raise ValueError("Invalid comparator in leaf.")
                    try:
                        thr = float(thr)
                    except Exception as e:
                        raise ValueError(f"Threshold not a float: {thr}") from e
                    if not (-2.0 <= thr <= 2.0):
                        raise ValueError(f"Threshold out of bounds: {thr}")
            else:
                raise ValueError("Malformed formula node (expected 3-item list).")
        for f in formulas[:k]:
            _check(f)
