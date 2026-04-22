import os
from .config import SYSTEM_PROMPT_PATH

# ---------------------------------------------------------------------------
# Load base system prompt from file
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROMPT_FILE = os.path.join(_BASE_DIR, SYSTEM_PROMPT_PATH)

with open(_PROMPT_FILE, "r", encoding="utf-8") as f:
    _BASE_SYSTEM_PROMPT = f.read()


_FEW_SHOT_EXAMPLES = """
## Examples of good responses

### Example 1 — Beginner user, no background in statistics
User: What is the mean?
Assistant:
The mean is just the average of a group of numbers.
**How to calculate it:** Add all the numbers together, then divide by how many numbers there are.
**Example:** For [2, 4, 6], the mean is (2 + 4 + 6) / 3 = 4.
**Why it matters:** It gives you a single number that summarises the centre of your data.

### Example 2 — Intermediate user, background in biology
User: What is the difference between variance and standard deviation?
Assistant:
Both measure how spread out your data is, but in different units.
**Variance** is the average squared distance from the mean. Squaring removes negative signs but also changes the units — if your data is in grams, variance is in grams².
**Standard deviation** is simply the square root of variance, which brings the units back to the original scale (grams), making it easier to interpret.
**In biology:** If you measured the weight of 50 mice, the standard deviation tells you directly "most mice were within ±X grams of the mean" — much more intuitive than variance.

### Example 3 — Advanced user, goal is to pass a machine learning exam
User: When should I use L1 vs L2 regularisation?
Assistant:
The choice depends on whether you expect sparsity in your feature set.
**L1 (Lasso):** Adds the sum of absolute weights as a penalty. Drives some weights exactly to zero, performing implicit feature selection. Prefer L1 when you believe only a subset of features are truly predictive.
**L2 (Ridge):** Adds the sum of squared weights. Shrinks all weights toward zero but rarely to exactly zero. Prefer L2 when most features contribute and you want to reduce multicollinearity.
**Exam tip:** Know that L1 produces sparse models (useful for interpretability) while L2 produces stable models (useful when features are correlated). ElasticNet combines both.
"""


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------
def build_system_prompt(
    user_level: str,
    user_goal: str,
    user_background: str,
) -> str:
    """
    Construct a personalised system prompt by injecting user profile fields
    into the base prompt template.

    Design decisions:
    - user_level controls vocabulary complexity and assumed prior knowledge
    - user_goal orients examples and emphasis toward what the user needs
    - user_background allows the model to use domain-relevant analogies
    - Few-shot examples are appended here (in the system prompt) so they
      apply globally across the whole conversation, not just one turn

    Args:
        user_level:      'Beginner', 'Intermediate', or 'Advanced'
        user_goal:       e.g. 'pass a university exam', 'get a data science job'
        user_background: e.g. 'biology', 'economics', 'no prior background'

    Returns:
        A fully composed system prompt string.
    """
    profile_block = f"""
## User Profile
- **Level:** {user_level}
- **Goal:** {user_goal}
- **Background:** {user_background}

## Personalisation Instructions
- Adapt your vocabulary and depth strictly to the user's level:
  - Beginner: use plain language, avoid jargon, always include a simple example
  - Intermediate: use correct terminology, assume basic maths literacy
  - Advanced: use precise technical language, include edge cases and tradeoffs
- Connect explanations to the user's background when relevant
  (e.g. if background is biology, use biological datasets as examples)
- Keep the user's goal in mind: if their goal is exam preparation, highlight
  exam-relevant distinctions; if it is industry, emphasise practical application
"""

    return _BASE_SYSTEM_PROMPT + "\n\n" + profile_block + "\n\n" + _FEW_SHOT_EXAMPLES


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------
def build_user_prompt(
    user_question: str,
    user_level: str,
    user_goal: str,
    user_background: str,
) -> str:
    """
    Compose the full user turn by combining the free-text question with
    chain-of-thought instructions and output format directives.

    Design decisions:
    - Chain-of-thought: asking the model to reason before answering improves
      accuracy on conceptual questions common in data science tutoring
    - Output format: structured responses are more useful for learning than
      free-form paragraphs — format adapts to user level via _get_format_directive
    - XML-like tags clearly delimit profile data from the question from
      instructions, reducing the risk of the model confusing them
    - Profile fields are repeated here (not just in the system prompt) to
      reinforce personalisation at the point of the actual question

    Args:
        user_question:   The raw question typed by the user
        user_level:      'Beginner', 'Intermediate', or 'Advanced'
        user_goal:       The user's stated learning goal
        user_background: The user's domain background

    Returns:
        A fully composed user turn string ready to send to the LLM.
    """
    format_directive = _get_format_directive(user_level)

    return f"""<user_profile>
Level: {user_level}
Goal: {user_goal}
Background: {user_background}
</user_profile>

<user_question>
{user_question}
</user_question>

<instructions>
Think step by step before writing your answer. First identify what concept
is being asked about and what the user needs to understand given their level
and goal. Then compose your response using this structure:

{format_directive}

Ensure your examples and analogies are relevant to the user's background
where possible.
</instructions>"""


def _get_format_directive(user_level: str) -> str:
    """
    Return a level-appropriate output format directive.
    Beginners benefit from a simple 3-part structure.
    Advanced users benefit from a more technical breakdown.
    """
    if user_level == "Beginner":
        return (
            "1. **Simple explanation** (1-2 sentences, no jargon)\n"
            "2. **Concrete example** (use numbers or a relatable scenario)\n"
            "3. **Why it matters** (one sentence on practical relevance)"
        )
    elif user_level == "Intermediate":
        return (
            "1. **Concept** (clear definition with correct terminology)\n"
            "2. **How it works** (brief explanation of the mechanism)\n"
            "3. **Example** (preferably from a relevant domain)\n"
            "4. **Common pitfall** (one mistake people make with this concept)"
        )
    else:  # Advanced
        return (
            "1. **Precise definition** (technical, no simplification)\n"
            "2. **Mechanism and assumptions** (when does this hold / break?)\n"
            "3. **Practical tradeoffs** (pros, cons, alternatives)\n"
            "4. **Exam / interview angle** (what is typically tested or asked)"
        )


# ---------------------------------------------------------------------------
# RAG context builder
# ---------------------------------------------------------------------------
def build_rag_context(chunks: list[str]) -> str:
    """
    Format retrieved text chunks into a context block for injection
    into the user message. Returns empty string if no chunks provided.
    """
    if not chunks:
        return ""
    
    numbered = ""
    for i, chunk in enumerate(chunks):
        numbered += f"[Source {i+1}]: {chunk}\n\n---\n\n"

    return (
        f"Use the following context to answer the question. "
        f"When you use information from a source, explicitly cite it as [Source 1], [Source 2], etc.\n\n"
        f"{numbered}"
    )