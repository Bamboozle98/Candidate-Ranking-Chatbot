import re
from pathlib import Path

import streamlit as st
import pandas as pd

# --- your existing scripts ---
from src.models.Default.features import add_basic_features
from src.models.Default.ranker import initial_filter, score_candidates
from src.models.Default.feedback import rerank_with_star

# Optional local text generation (free/offline)
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
            "I’m your ranking assistant. Try:\n"
            "- `rank`\n"
            "- `show top 10`\n"
            "- `star 123`\n"
            "- `rerank`\n"
            "- `set keywords: engineering manager`\n"
        )}
    ])

init_state()


# ----------------------------
# Data loading (static repo path)
# ----------------------------
st.sidebar.header("Ranking Controls")

REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "data" / "raw" / "potential-talents - Aspiring human resources - seeking human resources.csv"

# Your dataset may use either "connection" or "connections"
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


# ----------------------------
# Local model (cached) - optional
# ----------------------------
@st.cache_resource
def get_local_llm():
    # Free, local/offline. GPT-2 is small but works for simple text.
    return pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=120,
        temperature=0.3,
    )

def llm_explain(prompt: str) -> str:
    if not use_local_llm:
        return ""
    gen = get_local_llm()(prompt)
    # transformers pipeline returns list of dicts
    return gen[0]["generated_text"]


# ----------------------------
# Intent router (your “tool use”)
# ----------------------------
def parse_intent(text: str):
    t = text.strip()

    # set keywords: ....
    m = re.match(r"(?i)^\s*set\s+keywords\s*:\s*(.+)\s*$", t)
    if m:
        return ("set_keywords", m.group(1).strip())

    # star 123
    m = re.match(r"(?i)^\s*star\s+(\d+)\s*$", t)
    if m:
        return ("star", int(m.group(1)))

    # show top 10
    m = re.match(r"(?i)^\s*show\s+top\s+(\d+)\s*$", t)
    if m:
        return ("show", int(m.group(1)))

    # rank / rerank
    if re.search(r"(?i)\brerank\b", t):
        return ("rerank", None)

    if re.search(r"(?i)\brank\b", t):
        return ("rank", None)

    # fallback: chat/explain
    return ("chat", t)


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
            f"Explain in 1-2 sentences why starring a candidate helps rerank similar profiles."
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
    # keep ranked list but note it may now be stale
    return f"Keywords updated to: {new_kw}. Type `rank` to compute a new list."


def run_chat_turn(user_text: str) -> str:
    intent, payload = parse_intent(user_text)

    if intent == "set_keywords":
        return do_set_keywords(payload)
    if intent == "rank":
        return do_rank()
    if intent == "star":
        return do_star(payload)
    if intent == "rerank":
        return do_rerank()
    if intent == "show":
        return do_show(payload)

    # fallback chat: deterministic response (or GPT-2 if enabled)
    if use_local_llm:
        # Keep it very constrained—GPT-2 is not a reliable assistant
        prompt = (
            "You are a helpful assistant for a candidate ranking app. "
            "Respond briefly and suggest a command like: rank, show top 10, star 123, rerank.\n"
            f"User: {payload}\nAssistant:"
        )
        return llm_explain(prompt)

    return (
        "I can help with ranking commands. Try:\n"
        "- `rank`\n"
        "- `show top 10`\n"
        "- `star 123`\n"
        "- `rerank`\n"
        "- `set keywords: <text>`"
    )


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

    user_msg = st.chat_input("Try: 'rank', 'show top 10', 'star 123', 'rerank', 'set keywords: ...'")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            resp = run_chat_turn(user_msg)
            st.markdown(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp})
