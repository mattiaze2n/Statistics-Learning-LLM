import csv 
import os 
import time 
import streamlit as st 
from backend.llm_client import ask_stream

EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_set.csv")

def load_eval_set() -> list[dict]:
    "Load questions fromeval set"
    if not os.path.exists(EVAL_SET_PATH):
        return []
    with open(EVAL_SET_PATH, "r", encoding = "utf-8") as f:
        return list(csv.DictReader(f))
    

def save_eval_set(rows: list[dict]):
    """Write rows back to eval_set.csv"""
    if not rows:
        return 
    with open(EVAL_SET_PATH, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def render_eval_tab():
    """Render the evaluation tab"""
    st.subheader("Eval Mode")
    st.caption(
        "Run all tests questions against your current prompt"
        "Review answers to catch regressions before changing prompts"
    )

    eval_set = load_eval_set()

    with st.expander("Add a test question"):
        new_q = st.text_input("Question")
        new_topic = st.text_input("Exèected topic (what should the answer cover?)")
        new_level = st.selectbox("User elevel", ["Beginner", "Intermediate", "Advanced"], key="new_level")

        if st.button("Add to eval set") and new_q:
            eval_set.append({
                "question": new_q,
                "expected_topic": new_topic,
                "user_level": new_level,
            })
            save_eval_set(eval_set)
            st.success("Added.")
            st.rerun()

    if not eval_set:
        st.info(f"No eval questions yet. Add some above, or create {EVAL_SET_PATH}.")
        return 
        
    st.write(f"**{len(eval_set)} questions** in eval set.")

    if st.button("Run all questions", type = "primary"):
        results = []
        progress = st.progress(0, text = "Running eveluation...")


        for i, row in enumerate(eval_set):
            answer = "".join(list(ask_stream(
                user_message=row["question"],
                user_level=row.get("user_level", "Intermediate"),
                user_goal=st.session_state.get("user_goal", "learn data science"),
                user_background=st.session_state.get("user_background", "not specified"),
                conversation_history=[],   # fresh history for each eval question
            )))

            results.append({
                "question":       row["question"],
                "expected_topic": row.get("expected_topic", ""),
                "user_level":     row.get("user_level", ""),
                "answer":         answer,
            })
            progress.progress((i + 1) / len(eval_set), text=f"Question {i+1}/{len(eval_set)}")
            time.sleep(0.5)  # avoid rate limits
 
        st.session_state["eval_results"] = results
        progress.empty()
        st.success("Done. Review results below.")
        st.rerun()
 
    
    if "eval_results" in st.session_state:
        st.divider()
        st.subheader("Results")
        st.caption("Read each answer and mark it pass/fail. Your review is not saved automatically.")
 
        for i, result in enumerate(st.session_state["eval_results"]):
            with st.expander(
                f"Q{i+1} [{result['user_level']}] — {result['question'][:80]}",
                expanded=False,
            ):
                if result["expected_topic"]:
                    st.caption(f"Expected to cover: **{result['expected_topic']}**")
                st.markdown(result["answer"])
 
                col1, col2 = st.columns(2)
                with col1:
                    st.button("✅ Pass", key=f"pass_{i}")
                with col2:
                    st.button("❌ Fail", key=f"fail_{i}")
