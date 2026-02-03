import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import streamlit as st
import pandas as pd

# --- your existing scripts ---
from src.models.Default.features import add_basic_features
from src.models.Default.ranker import initial_filter, score_candidates
from src.models.Default.feedback import rerank_with_star

# Optional local models (free/offline)
from transformers import pipeline


# ----------------------------
# Streamlit setup + state
# ----------------------------
st.set_page_config(page_title="Candidate Ranker + Free Chat", layout="wide")

def init_state():
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("keywords", "aspiring human resources")
    st.session_state.setdefault("starred_id", None)
    st.session_state.setdefault("ranked", None)
    st.session_state.setdefault("messages", [
        {"role": "assistant", "content": (
            "I’m your ranking assistant. You can type natural language, or commands like:\n"
            "- `rank`\n"
            "- `show top 10`\n"
            "- `star 123`\n"
            "- `rerank`\n"
            "- `set keywords: engineering manager`\n"
            "\nExamples of natural language:\n"
            "- “Rank candidates for engineering manager”\n"
            "- “Star candidate 7 and rerank”\n"
            "- “Show me the top 15”\n"
        )}
    ])

init_state()


# ----------------------------
# Data loading (static repo path)
# ----------------------------
st.sidebar.header("Ranking Controls")

REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "data" / "raw" / "potential-talents - Aspiring human resources - seeking human resources.csv"

REQUIRED_COLS = {"id", "job_title", "location"}  # handle connection separately

@st.cache_data
def load_candidates(csv_path: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_path)

    missing = REQUIRED_COLS - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Normalize connection column name to match your pipeline expectations
    if "connections" in df_raw.columns and "connection" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"connections": "connection"})

    if "connection" not in df_raw.columns:
        raise ValueError("CSV must contain 'connection' or 'connections' column.")

    return add_basic_features(df_raw)

try:
    st.session_state.df = load_candidates(CSV_PATH)
    st.sidebar.success(f"Loaded {len(st.session_state.df)} candidates from {CSV_PATH.name}")
except FileNotFoundError:
    st.session_state.df = None
    st.sidebar.error(f"CSV not found at: {CSV_PATH}")
except Exception as e:
    st.session_state.df = None
    st.sidebar.error(f"Failed to load CSV: {e}")

# Controls
st.sidebar.text_input("Role keywords", key="keywords")

alpha = st.sidebar.slider(
    "Star influence (alpha)",
    0.0, 1.0, 0.4, 0.05,
    help="Lower alpha = more like starred candidate; higher alpha = more baseline score."
)

filter_tau = st.sidebar.slider(
    "Relevance filter threshold (tau)",
    0.0, 0.5, 0.08, 0.01,
    help="Candidates below this keyword similarity are filtered out."
)

top_k = st.sidebar.slider("Top K to display", 5, 100, 25, 5)

use_local_llm = st.sidebar.checkbox(
    "Use local GPT-2 for explanations (optional)",
    value=False,
    help="If off, responses are purely deterministic. If on, GPT-2 adds natural language explanations."
)

use_instruction_llm = st.sidebar.checkbox(
    "Use local instruction model for natural language commands (recommended)",
    value=True,
    help="If on, the app will extract actions from natural language using a free local model."
)


# ----------------------------
# Local models (cached)
# ----------------------------
@st.cache_resource
def get_explainer_llm():
    # Free, local/offline. GPT-2 is small but OK for light explanations.
    return pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=120,
        temperature=0.3,
    )


def llm_explain(prompt: str) -> str:
    if not use_local_llm:
        return ""
    gen = get_explainer_llm()(prompt)
    return gen[0]["generated_text"]


@st.cache_resource
def get_instruction_model():
    # Instruction-tuned, better than GPT-2 for command extraction (free/offline).
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=128,
    )


# ----------------------------
# Deterministic “tools” (no LangChain)
# ----------------------------
def do_rank():
    if st.session_state.df is None:
        return "No dataset loaded."

    df = st.session_state.df
    kw = st.session_state.keywords

    scored = score_candidates(df, kw, model_path=None)
    filtered = initial_filter(scored, kw, tau=filter_tau)

    st.session_state.ranked = filtered
    st.session_state.starred_id = None

    msg = f"Ranked {len(filtered)} candidates for keywords='{kw}'."
    if use_local_llm:
        msg += "\n\n" + llm_explain(
            f"Explain briefly what it means to rank candidates by keyword relevance. Keywords: {kw}"
        )
    return msg

def do_star(candidate_id: int):
    if st.session_state.ranked is None:
        return "No ranking exists yet. Type `rank` first."

    if candidate_id not in set(st.session_state.ranked["id"]):
        return f"Candidate id={candidate_id} not found in the current ranked list."

    st.session_state.starred_id = candidate_id
    msg = f"Starred candidate id={candidate_id}. Type `rerank` to update the list."
    if use_local_llm:
        msg += "\n\n" + llm_explain(
            "Explain in 1-2 sentences why starring a candidate helps rerank similar profiles."
        )
    return msg

def do_rerank():
    if st.session_state.ranked is None:
        return "No ranking exists yet. Type `rank` first."
    if st.session_state.starred_id is None:
        return "No starred candidate yet. Type `star <id>` first."

    try:
        st.session_state.ranked = rerank_with_star(
            st.session_state.ranked,
            starred_id=st.session_state.starred_id,
            keywords=st.session_state.keywords,
            alpha=alpha
        )
        msg = f"Re-ranked using starred_id={st.session_state.starred_id} (alpha={alpha})."
        if use_local_llm:
            msg += "\n\n" + llm_explain(
                "Briefly explain how blending baseline score and similarity-to-star changes ordering."
            )
        return msg
    except Exception as e:
        return f"Rerank failed: {e}"

def do_show(n: int):
    if st.session_state.ranked is None:
        return "No ranking exists yet. Type `rank` first."

    n = int(max(1, min(50, n)))
    df = st.session_state.ranked.head(n).copy()

    cols = ["id", "job_title", "location", "connection"]
    score_cols = [c for c in ["final_score", "base_fit", "kw_sim", "star_sim"] if c in df.columns]
    cols = cols + score_cols

    return df[cols].to_markdown(index=False)

def do_set_keywords(new_kw: str):
    st.session_state.keywords = new_kw
    st.session_state.starred_id = None
    return f"Keywords updated to: {new_kw}. Type `rank` to compute a new list."


# ----------------------------
# Command extraction (rules + instruction model fallback)
# ----------------------------
ALLOWED_ACTIONS = {"rank", "rerank", "star", "show_top", "set_keywords"}

def rule_parse_command(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()

    m = re.match(r"(?i)^\s*set\s+keywords\s*:\s*(.+)\s*$", t)
    if m:
        return {"action": "set_keywords", "keywords": m.group(1).strip()}

    m = re.match(r"(?i)^\s*star\s+(\d+)\s*$", t)
    if m:
        return {"action": "star", "candidate_id": int(m.group(1))}

    m = re.match(r"(?i)^\s*show\s+top\s+(\d+)\s*$", t)
    if m:
        return {"action": "show_top", "n": int(m.group(1))}

    if re.search(r"(?i)\brerank\b", t):
        return {"action": "rerank"}

    if re.search(r"(?i)\brank\b", t):
        return {"action": "rank"}

    return None

def validate_command(cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(cmd, dict):
        return None

    action = cmd.get("action")
    if action not in ALLOWED_ACTIONS:
        return None

    if action == "star":
        cid = cmd.get("candidate_id")
        if not isinstance(cid, int):
            return None
        return {"action": "star", "candidate_id": cid}

    if action == "show_top":
        n = cmd.get("n")
        if not isinstance(n, int):
            return None
        n = max(1, min(50, n))
        return {"action": "show_top", "n": n}

    if action == "set_keywords":
        kw = cmd.get("keywords")
        if not isinstance(kw, str) or not kw.strip():
            return None
        return {"action": "set_keywords", "keywords": kw.strip()}

    return {"action": action}

def llm_parse_command(text: str) -> Optional[Dict[str, Any]]:
    if not use_instruction_llm:
        return None

    gen = get_instruction_model()

    prompt = (
        "You are a command parser for a candidate ranking app.\n"
        "Convert the user's message into EXACTLY one JSON object and nothing else.\n"
        "Allowed actions:\n"
        "- rank\n"
        "- rerank\n"
        "- star (requires candidate_id integer)\n"
        "- show_top (requires n integer 1-50)\n"
        "- set_keywords (requires keywords string)\n"
        "\n"
        f"User message:\n{text}\n\n"
        "Return only JSON:\n"
    )

    out = gen(prompt)[0]["generated_text"].strip()

    # Extract a JSON object defensively
    m = re.search(r"\{.*\}", out, flags=re.DOTALL)
    if not m:
        return None

    try:
        cmd = json.loads(m.group(0))
    except Exception:
        return None

    return validate_command(cmd)

def extract_command(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    # 1) fast rules
    cmd = rule_parse_command(text)
    if cmd:
        cmd = validate_command(cmd)
        if cmd:
            return cmd, "rules"

    # 2) instruction model fallback
    cmd = llm_parse_command(text)
    if cmd:
        return cmd, "llm"

    return None, "none"


# ----------------------------
# Chat handling
# ----------------------------
def run_chat_turn(user_text: str) -> str:
    cmd, source = extract_command(user_text)

    if cmd is None:
        # fallback “chat” behavior
        if use_local_llm:
            prompt = (
                "You are a helpful assistant for a candidate ranking app. "
                "Respond briefly and suggest a command like: rank, show top 10, star 123, rerank, set keywords: ...\n"
                f"User: {user_text}\nAssistant:"
            )
            return llm_explain(prompt)

        return (
            "I didn’t understand that. Try:\n"
            "- `rank`\n"
            "- `show top 10`\n"
            "- `star 123`\n"
            "- `rerank`\n"
            "- `set keywords: <text>`\n"
            "Or natural language like: “Show me the top 15”"
        )

    action = cmd["action"]

    if action == "set_keywords":
        return do_set_keywords(cmd["keywords"])

    if action == "rank":
        # If user typed natural language like “rank for data scientist”, try to also update keywords
        # (Optional light enhancement)
        return do_rank()

    if action == "star":
        return do_star(cmd["candidate_id"])

    if action == "rerank":
        return do_rerank()

    if action == "show_top":
        return do_show(cmd["n"])

    return "Unsupported action (this should not happen)."


# ----------------------------
# UI layout
# ----------------------------
st.title("Candidate Ranking + Chatbot (Free / Offline)")

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Ranked Candidates")
    if st.session_state.ranked is None:
        st.info("No ranking yet. Use chat: `rank`.")
    else:
        view = st.session_state.ranked.head(top_k).copy()
        base_cols = ["id", "job_title", "location", "connection"]
        score_cols = [c for c in ["final_score", "base_fit", "kw_sim", "star_sim"] if c in view.columns]
        st.dataframe(view[base_cols + score_cols], use_container_width=True)

        st.caption(
            f"keywords='{st.session_state.keywords}' | starred_id={st.session_state.starred_id} | "
            f"tau={filter_tau} | alpha={alpha}"
        )

with right:
    st.subheader("Chat")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Try natural language: “Show me the top 10”, “Star 7 and rerank”, “Rank for HR manager”")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            resp = run_chat_turn(user_msg)
            st.markdown(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp})
