import json
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.models.Default.features import add_basic_features
from src.models.Default.ranker import initial_filter, score_candidates
from src.models.Default.feedback import rerank_with_star, save_star

# ---------------------------
# Config
# ---------------------------
CSV_PATH = r"C:\Users\cbran\PycharmProjects\8XPTuDF1AleElmm6\data\raw\potential-talents - Aspiring human resources - seeking human resources.csv"
STAR_PATH = r"C:\Users\cbran\PycharmProjects\8XPTuDF1AleElmm6\src\models\Default\artifacts\starred.json"

# Reference Ollama model
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:3b-instruct"

# Order ranking priority
ID_COL_PRIORITY = ["id", "candidate_id", "CandidateID", "candidateId"]
SCORE_COL_PRIORITY = ["score", "base_fit", "fit", "kw_sim", "final_score", "rank_score"]


# ---------------------------
# Ollama helpers
# ---------------------------
def ollama_chat(messages: List[Dict[str, str]], timeout_s: int = 120) -> str:
    payload = {"model": MODEL, "messages": messages, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Ollama not reachable at {OLLAMA_URL}. "
            f"Make sure Ollama is running and the model is pulled. "
            f"Original error: {e}"
        )


# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PLANNER = """You are a local recruiting assistant that routes requests to tools.
Output format rules:
- First line: a short greeting (one line).
- Then output ONLY one JSON object (no markdown, no extra text).

JSON schema:
{
  "job_title": string,
  "instruction": string,
  "filters": object,
  "top_k": integer,
  "do_rerank": boolean
}

Guidelines:
- If the user did not provide a job title, set "job_title" to "".
- Put hard constraints into filters (e.g., location, min_years, visa_required=false, remote=true).
- Put softer preferences into instruction (e.g., "prioritize healthcare + Python").
- Never invent candidates. You are only planning.
"""

SYSTEM_FORMATTER = """You are a helpful assistant in a candidate ranking app.

Rules:
- Do NOT print or restate the ranked list (IDs/scores). The UI already shows it.
- Do NOT invent any candidate attributes.
- Write 1–3 short sentences:
  1) confirm what you did (rank/rerank/show) using the given job_title and any constraints
  2) if rerank: mention the starred_id used

If tool_result.ok is false, explain the error and what the user should do next.
"""



REQUIRED_KEYS = {"job_title", "instruction", "filters", "top_k", "do_rerank"}


# ---------------------------
# Plan parsing / validation
# ---------------------------
def extract_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"LLM did not return a JSON object. Got: {text[:200]}")
    return json.loads(text[start:end + 1])


def validate_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    missing = REQUIRED_KEYS - set(plan.keys())
    if missing:
        raise ValueError(f"Plan missing keys: {missing}")

    if not isinstance(plan["job_title"], str):
        raise ValueError("job_title must be string")
    if not isinstance(plan["instruction"], str):
        raise ValueError("instruction must be string")
    if not isinstance(plan["filters"], dict):
        raise ValueError("filters must be object/dict")
    if not isinstance(plan["top_k"], int) or plan["top_k"] <= 0:
        raise ValueError("top_k must be positive integer")
    if not isinstance(plan["do_rerank"], bool):
        raise ValueError("do_rerank must be boolean")

    plan["top_k"] = min(plan["top_k"], 200)
    return plan


# ---------------------------
# Ranking helpers from other files (features.py, feedback.py, ranker.py)
# ---------------------------
def job_title_to_keywords(job_title: str) -> str:
    return job_title.strip()


def pick_column(df: pd.DataFrame, candidates: List[str], kind: str) -> str:
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(
            f"Could not find {kind} column. Expected one of {candidates}. "
            f"Got columns: {list(df.columns)}"
        )
    return col


def normalize_scored_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Ensures:
      - there is an 'id' column
      - there is a 'score' column
    Returns (df_norm, id_col, score_col) where id_col will be 'id' and score_col will be 'score'.
    """
    df = df.copy()

    # ID
    id_col = next((c for c in ID_COL_PRIORITY if c in df.columns), None)
    if id_col is None:
        df = df.reset_index(drop=False).rename(columns={"index": "id"})
        id_col = "id"

    # Score
    raw_score_col = next((c for c in SCORE_COL_PRIORITY if c in df.columns), None)
    if raw_score_col is None:
        raise ValueError(
            f"No known score column found. Expected one of {SCORE_COL_PRIORITY}. "
            f"Got columns: {list(df.columns)}"
        )

    if raw_score_col != "score":
        df["score"] = df[raw_score_col]
    score_col = "score"

    # Canonical id column name for downstream
    if id_col != "id":
        df["id"] = df[id_col]
        id_col = "id"

    return df, id_col, score_col


def df_to_results(df: pd.DataFrame, top_k: int) -> list[dict]:
    top = df.head(top_k)
    return [{"id": int(r["id"]), "score": float(r["score"])} for _, r in top.iterrows()]



# ---------------------------
# Core pipeline based on the ranking scripts
# ---------------------------
def rank_candidates(job_title: str, filters: Dict[str, Any], top_k: int) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    keywords = job_title_to_keywords(job_title)

    df = pd.read_csv(CSV_PATH)
    df = add_basic_features(df)

    # 1) filter
    df_f = initial_filter(df, keywords, tau=0.08)

    # 2) score (KEEP all original columns)
    df_scored = score_candidates(df_f, keywords, model_path=None)

    # 3) normalize (may drop columns) -> merge score back into df_scored
    df_norm, _, _ = normalize_scored_df(df_scored)
    df_scored = df_scored.merge(df_norm[["id", "score"]], on="id", how="left")

    # 4) sort + return
    df_scored = df_scored.sort_values("score", ascending=False)

    results = df_to_results(df_scored, top_k=top_k)  # should pick id+score
    return results, df_scored


def rerank_candidates(job_title: str, df_scored: pd.DataFrame, starred_id: int):
    keywords = job_title_to_keywords(job_title)

    save_star(STAR_PATH, keywords, int(starred_id))

    df_r_full = rerank_with_star(
        df_scored,
        starred_id=int(starred_id),
        keywords=keywords,
        alpha=0.4
    )

    # ---- Ensure we have a score input column ----
    # If rerank_with_star already produced "score", great.
    # Otherwise map whichever column it produced into "score".
    if "score" not in df_r_full.columns:
        # Common patterns — adjust if your rerank_with_star uses a different name
        if "rerank_score" in df_r_full.columns:
            df_r_full["score"] = df_r_full["rerank_score"]
        elif "final_score" in df_r_full.columns:
            df_r_full["score"] = df_r_full["final_score"]
        elif "base_fit" in df_r_full.columns:
            # fallback: use pre-existing score signal
            df_r_full["score"] = df_r_full["base_fit"]
        else:
            raise ValueError(f"rerank_with_star did not output a score column. Columns: {list(df_r_full.columns)}")

    # Normalize (if normalize_scored_df also renames/overwrites score, that's fine)
    df_norm, _, _ = normalize_scored_df(df_r_full)

    # Merge normalized score back to keep full candidate info
    df_r_full = df_r_full.drop(columns=["score"], errors="ignore").merge(
        df_norm[["id", "score"]],
        on="id",
        how="left"
    )

    df_r_full = df_r_full.sort_values("score", ascending=False)
    results = df_to_results(df_r_full, top_k=len(df_r_full))
    return results, df_r_full




# ---------------------------
# Assistant wrapper
# ---------------------------
class CandidateRankAssistant:
    def __init__(self):
        self.last_job_title: str = ""
        self.last_df_scored: Optional[pd.DataFrame] = None
        self.last_results: Optional[List[Dict[str, Any]]] = None

    def handle(self, user_text: str, starred_id: Optional[int] = None) -> Dict[str, Any]:
        # Step 1: Plan
        plan_text = ollama_chat([
            {"role": "system", "content": SYSTEM_PLANNER},
            {"role": "user", "content": user_text},
        ])
        plan = validate_plan(extract_json_object(plan_text))

        # Memory: allow “rerank it” without repeating title
        if not plan["job_title"].strip() and self.last_job_title:
            plan["job_title"] = self.last_job_title

        if not plan["job_title"].strip():
            return {"reply": "Hi! What job title should I rank candidates for?", "plan": plan, "results": None}

        self.last_job_title = plan["job_title"].strip()

        # Step 2: Rank
        results, df_scored = rank_candidates(plan["job_title"], plan["filters"], plan["top_k"])
        self.last_results = results
        self.last_df_scored = df_scored

        final_list = results

        # Step 2b: Rerank (requires UI to pass starred_id from user)
        if plan["do_rerank"]:
            if starred_id is None:
                return {
                    "reply": "I can rerank, but I need a starred candidate_id (pick one from the results).",
                    "plan": plan,
                    "results": results
                }

            final_list, df_reranked = rerank_candidates(plan["job_title"], df_scored, starred_id)
            self.last_results = final_list
            self.last_df_scored = df_reranked

        # Step 3: Format (LLM writes only; ordering already final)
        formatted = ollama_chat([
            {"role": "system", "content": SYSTEM_FORMATTER},
            {"role": "user", "content": json.dumps({
                "job_title": plan["job_title"],
                "instruction": plan["instruction"],
                "filters": plan["filters"],
                "candidates": final_list[:10],
            })}
        ])

        return {"reply": formatted, "plan": plan, "results": final_list}


# ---------------------------
# CLI demo
# ---------------------------
if __name__ == "__main__":
    bot = CandidateRankAssistant()

    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        out = bot.handle(user)
        print("\nAssistant:\n" + out["reply"])

        if out["plan"]["do_rerank"] and out["results"]:
            try:
                starred = int(input("\nStar which candidate_id? ").strip())
                out2 = bot.handle(user, starred_id=starred)
                print("\nAssistant (reranked):\n" + out2["reply"])
            except Exception as e:
                print("Rerank failed:", e)
