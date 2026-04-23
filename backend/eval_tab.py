import csv 
import os 
import time 
import streamlit as st 
import json
from datetime import datetime
from backend.llm_client import ask_stream, ask

EVAL_SET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_set.csv")
RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "eval_results_history.csv"
)

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

def evaluate_answer(
    question: str,
    answer: str,
    expected_topic: str,
    user_level: str,
    user_background: str,
) -> dict:
    """
    Use the LLM to evaluate an answer against a rubric.
    Returns a dict with scores and feedback.
    """


    eval_prompt = f"""You are an expert statistics educator evaluating a tutoring response.

Question asked: {question}
Expected topic: {expected_topic}
User level: {user_level}
User background: {user_background}

Response to evaluate:
{answer}

Evaluate the response on these four dimensions. For each, give a score 1-3 and one sentence of feedback.
1 = poor, 2 = acceptable, 3 = good.

Respond ONLY in JSON with this exact structure:
{{
    "correctness": {{"score": 1-3, "feedback": "..."}},
    "personalisation": {{"score": 1-3, "feedback": "..."}},
    "structure": {{"score": 1-3, "feedback": "..."}},
    "background_relevance": {{"score": 1-3, "feedback": "..."}}
}}

Correctness: is the statistical content accurate?
Personalisation: is the depth and vocabulary appropriate for the user level?
Structure: does it follow a clear format with explanation, example, and relevance?
Background relevance: does it use examples relevant to the user background?
"""

    response = ask(
        user_message=eval_prompt,
        user_level="Advanced",
        user_goal="evaluate a response",
        user_background="statistics education",
        conversation_history=[],
    )

    if "error" in response:
        return {"error": response["error"]}

    try:
        import json
        clean = response["answer"].replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Could not parse evaluation response"}

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
        new_topic = st.text_input("Expected topic...")
        new_level = new_level = st.selectbox("User level", ["Beginner", "Intermediate", "Advanced"], key="new_level")

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
        progress = st.progress(0, text = "Running evaluation...")

        for i, row in enumerate(eval_set):
            answer = "".join(list(ask_stream(
                    user_message=row["question"],
                    user_level=row.get("user_level", "Intermediate"),
                    user_goal=row.get("user_goal", "learn data science"),
                    user_background=row.get("user_background", "not specified"),
                    conversation_history=[],
            )))

            evaluation = evaluate_answer(
                    question=row["question"],
                    answer=answer,
                    expected_topic=row.get("expected_topic", ""),
                    user_level=row.get("user_level", "Intermediate"),
                    user_background=row.get("user_background", "not specified"),
            )

            results.append({
                    "question":        row["question"],
                    "expected_topic":  row.get("expected_topic", ""),
                    "user_level":      row.get("user_level", ""),
                    "user_goal":       row.get("user_goal", ""),
                    "user_background": row.get("user_background", ""),
                    "answer":          answer,
                    "evaluation":      evaluation,
            })

            progress.progress((i + 1) / len(eval_set), text=f"Question {i+1}/{len(eval_set)}")
            time.sleep(0.5)
            
        st.session_state["eval_results"] = results
        save_results_to_history(results)
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

                evaluation = result.get("evaluation", {})
                if evaluation and "error" not in evaluation:
                    st.divider()
                    st.caption("Automated evaluation:")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    dims = {
                        "Correctness":   "correctness",
                        "Personalisation": "personalisation", 
                        "Structure":     "structure",
                        "Background":    "background_relevance",
                    }
                    
                    for col, (label, key) in zip([col1, col2, col3, col4], dims.items()):
                        with col:
                            score = evaluation.get(key, {}).get("score", 0)
                            feedback = evaluation.get(key, {}).get("feedback", "")
                            color = "green" if score == 3 else "orange" if score == 2 else "red"
                            st.markdown(f"**{label}**")
                            st.markdown(f":{color}[{'★' * score}{'☆' * (3-score)}]")
                            st.caption(feedback)
                elif "error" in evaluation:
                    st.caption(f"Evaluation failed: {evaluation['error']}")
 
                col1, col2 = st.columns(2)
                with col1:
                    st.button("✅ Pass", key=f"pass_{i}")
                with col2:
                    st.button("❌ Fail", key=f"fail_{i}")


def save_results_to_history(results: list[dict]):
    """Append eval results to a history CSV with a timestamp."""
    if not results:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    rows = []
    for r in results:
        evaluation = r.get("evaluation", {})
        rows.append({
            "timestamp":              timestamp,
            "question":               r["question"],
            "user_level":             r.get("user_level", ""),
            "user_goal":              r.get("user_goal", ""),
            "user_background":        r.get("user_background", ""),
            "expected_topic":         r.get("expected_topic", ""),
            "answer":                 r["answer"],
            "correctness_score":      evaluation.get("correctness", {}).get("score", ""),
            "correctness_feedback":   evaluation.get("correctness", {}).get("feedback", ""),
            "personalisation_score":  evaluation.get("personalisation", {}).get("score", ""),
            "personalisation_feedback": evaluation.get("personalisation", {}).get("feedback", ""),
            "structure_score":        evaluation.get("structure", {}).get("score", ""),
            "structure_feedback":     evaluation.get("structure", {}).get("feedback", ""),
            "background_score":       evaluation.get("background_relevance", {}).get("score", ""),
            "background_feedback":    evaluation.get("background_relevance", {}).get("feedback", ""),
        })

    file_exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)