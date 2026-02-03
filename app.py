import json
import streamlit as st
import time
import pandas as pd
import re

from src.models.Default.assistant import (
    ollama_chat,
    rank_candidates,
    rerank_candidates,
    SYSTEM_FORMATTER,   # adjust formatter in assistant.py as needed.
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "snapshots" not in st.session_state:
    st.session_state.snapshots = []

if "memory" not in st.session_state:
    st.session_state.memory = {"job_title": "", "results": None, "df_scored": None}

TOOL_SCHEMA = """
You are a tool-router.
Return ONLY valid JSON. No markdown. No extra text. No explanation.

Output must match ONE of these exactly:

{"tool":"rank","args":{"job_title":"...","filters":{},"top_k":25}}
{"tool":"rerank","args":{"starred_id":123}}
{"tool":"show","args":{"top_n":10}}
{"tool":"set_job","args":{"job_title":"..."}}
{"tool":"help","args":{}}

Rules:
- Use double quotes for all keys/strings.
- args must always exist (even if empty).
- Do not include any other keys.
- If unsure, return {"tool":"help","args":{}}.
- top_k must be an int 1..200
- top_n must be an int 1..50
"""

DISPLAY_COLS = ["id", "job_title", "location", "connection", "fit"]


def build_display_df(results: list[dict], df_scored: pd.DataFrame) -> pd.DataFrame | None:
    """
    results: list like [{"candidate_id": 83, "score": 0.45}, ...] OR [{"id": 83, "score": ...}]
    df_scored: dataframe with full candidate columns, including 'id'
    """
    if df_scored is None or not results:
        return None

    r = pd.DataFrame(results)

    # Normalize the id field name
    if "id" not in r.columns and "candidate_id" in r.columns:
        r = r.rename(columns={"candidate_id": "id"})

    if "id" not in r.columns:
        raise ValueError("Ranking results must include 'id' (or 'candidate_id').")

    if "id" not in df_scored.columns:
        raise ValueError("df_scored must include an 'id' column to join on.")

    # Merge so we get all candidate columns
    merged = r.merge(df_scored, on="id", how="left")

    # Add explicit rank column (1..N)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    # Put the important columns up front (only keep those that exist)
    front = ["rank"] + [c for c in DISPLAY_COLS if c in merged.columns]
    # Keep score near the front if present
    if "score" in merged.columns:
        front = ["rank", "id", "score"] + [c for c in DISPLAY_COLS if c not in ("id",) and c in merged.columns]

    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    return merged


def extract_int_id(text: str) -> int | None:
    m = re.search(r"\b(\d{1,9})\b", text)
    return int(m.group(1)) if m else None


def save_snapshot(tool_name: str, job_title: str, results: list[dict], label: str = "") -> int:
    st.session_state.snapshots.append({
        "ts": time.time(),
        "label": label or tool_name,
        "tool": tool_name,
        "job_title": job_title,
        "results": results[:10],  # store top rows for table display
    })
    return len(st.session_state.snapshots) - 1


def safe_parse_json(text: str) -> dict:
    # model must output JSON only
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    return json.loads(text[start:end+1])

def tool_router(user_text: str, memory: dict) -> dict:
    # Provide a tiny state summary to help tool decisions
    state_summary = {
        "known_job_title": memory.get("job_title", ""),
        "have_results": memory.get("results") is not None,
        "num_results": len(memory["results"]) if memory.get("results") else 0,
    }
    raw = ollama_chat([
        {"role":"system","content":TOOL_SCHEMA},
        {"role":"user","content":json.dumps({"user_text": user_text, "state": state_summary})}
    ])
    return safe_parse_json(raw)

def assistant_reply(user_text: str, tool_name: str, tool_result: dict) -> str:
    # The assistant can write naturally but can’t invent candidate details. Assistant settings are in the FORMATTER variable in the assistant.py
    payload = {
        "user_text": user_text,
        "tool": tool_name,
        "tool_result": tool_result,
    }
    return ollama_chat([
        {"role":"system","content":SYSTEM_FORMATTER},
        {"role":"user","content":json.dumps(payload)}
    ])

st.set_page_config(page_title="Candidate Rank Chatbot", layout="wide")
st.title("Candidate Rank Chatbot (Tools + Ollama)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = {"job_title": "", "results": None, "df_scored": None}

# render chat history
# render chat history (ONLY ONCE)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

        if m["role"] == "assistant" and m.get("snapshot_idx") is not None:
            snap = st.session_state.snapshots[m["snapshot_idx"]]
            st.caption(f"{snap['label']} • {snap['job_title']}")
            st.dataframe(snap["results"], use_container_width=True)


prompt = st.chat_input("Ask me to rank candidates, star an id, rerank, show top N...")

if prompt:
    # show user message
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # decide tool
    try:
        action = tool_router(prompt, st.session_state.memory)
        # This is a function to display tool calling for debugging purposes
        # st.sidebar.json(action)
    except Exception as e:
        err = f"Tool routing failed: {e}"
        st.session_state.messages.append({"role":"assistant","content":err})
        with st.chat_message("assistant"):
            st.error(err)
        st.stop()

    tool = action.get("tool")
    args = action.get("args", {})

    tool_result = {"ok": True, "tool": tool, "args": args}

    # execute tool
    try:
        if tool == "help":
            tool_result["ok"] = True
            tool_result["message"] = (
                "Tell me a job title to rank for (e.g., 'HR Generalist') and optionally constraints "
                "(location, remote, etc.). You can also say 'show top 10' or 'rerank star 123'."
            )

        elif tool == "set_job":
            st.session_state.memory["job_title"] = args.get("job_title","").strip()
            tool_result["job_title"] = st.session_state.memory["job_title"]

        elif tool == "rank":
            job_title = args.get("job_title") or st.session_state.memory.get("job_title","")
            job_title = (job_title or "").strip()
            if not job_title:
                tool = "help"
                tool_result = {"ok": True, "tool":"help", "message":"What job title should I rank for?"}
            else:
                filters = args.get("filters", {}) or {}
                top_k = int(args.get("top_k", 25))
                top_k = min(max(top_k, 1), 200)

                results, df_scored = rank_candidates(job_title, filters, top_k)
                st.session_state.memory["job_title"] = job_title
                st.session_state.memory["results"] = results
                st.session_state.memory["df_scored"] = df_scored

                tool_result["job_title"] = job_title
                tool_result["num_results"] = len(results)
                tool_result["top10"] = results[:10]
                save_snapshot("rank", job_title, results, label=f"rank top {top_k}")

        elif tool == "show":
            if not st.session_state.memory.get("results"):
                tool_result = {"ok": False, "error": "No results yet. Ask me to rank first."}
            else:
                top_n = int(args.get("top_n", 10))
                top_n = min(max(top_n, 1), 50)
                tool_result["top"] = st.session_state.memory["results"][:top_n]

        elif tool == "rerank":
            n = extract_int_id(prompt)
            if n is not None:
                args["starred_id"] = n
            if "score" not in st.session_state.memory["df_scored"].columns:
                tool_result = {"ok": False,
                               "error": "Cannot rerank because df_scored has no 'score' column. Rank first (or fix scoring pipeline)."}
            else:
                starred_id = int(args.get("starred_id"))
                ...

            if st.session_state.memory.get("df_scored") is None:
                tool_result = {"ok": False, "error": "No scored dataframe yet. Rank first."}
            else:
                starred_id = int(args.get("starred_id"))
                job_title = st.session_state.memory.get("job_title","")
                results, df_r = rerank_candidates(job_title, st.session_state.memory["df_scored"], starred_id)
                st.session_state.memory["results"] = results
                st.session_state.memory["df_scored"] = df_r
                tool_result["starred_id"] = starred_id
                tool_result["top10"] = results[:10]
                save_snapshot("rerank", job_title, results, label=f"rerank star {starred_id}")

        else:
            tool_result = {"ok": False, "error": f"Unknown tool: {tool}"}

    except Exception as e:
        tool_result = {"ok": False, "error": str(e), "tool": tool, "args": args}

    # ----------------------------
    # Assistant output (NO debug JSON, NO double render)
    # ----------------------------

    # Create a snapshot only when we have something table-worthy
    snap_idx = None

    if tool in ("rank", "rerank") and tool_result.get("ok") and st.session_state.memory.get("results"):
        display_df = build_display_df(
            st.session_state.memory["results"],
            st.session_state.memory.get("df_scored"),
        )

        if display_df is not None:
            snap_idx = save_snapshot(
                tool_name=tool,
                job_title=st.session_state.memory.get("job_title", ""),
                results=display_df.to_dict("records"),
                label=f"{tool} • {st.session_state.memory.get('job_title', '')}"
            )

    elif tool == "show" and tool_result.get("ok") and tool_result.get("top"):
        display_df = build_display_df(tool_result["top"], st.session_state.memory.get("df_scored"))

        if display_df is not None:
            snap_idx = save_snapshot(
                tool_name="show",
                job_title=st.session_state.memory.get("job_title", ""),
                results=display_df.to_dict("records"),
                label=f"show top {len(tool_result['top'])}"
            )

    # Now produce assistant text (keep it short; tables show details)
    with st.chat_message("assistant"):
        try:
            reply = assistant_reply(prompt, tool, tool_result)
        except Exception as e:
            reply = f"Sorry — I hit an error formatting the response: {e}"

        st.markdown(reply)

        # If we made a snapshot, show it right now
        if snap_idx is not None:
            snap = st.session_state.snapshots[snap_idx]
            st.caption(f"{snap['label']}")
            st.dataframe(snap["results"], use_container_width=True)

    # Append exactly ONCE (this is what fixes double-printing)
    msg = {"role": "assistant", "content": reply}
    if snap_idx is not None:
        msg["snapshot_idx"] = snap_idx
    st.session_state.messages.append(msg)

