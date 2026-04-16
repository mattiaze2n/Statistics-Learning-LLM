"""
Tests for backend.llm_client and backend.prompts.

Run with:
    pytest tests/test_llm_client.py -v

No real API calls are made — the LangChain chain is mocked entirely.
"""

from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_SYSTEM_PROMPT = "You are a helpful data science tutor."
FAKE_LLM_ANSWER = "Variance measures how spread out numbers are."


def make_ask(mock_answer: str = FAKE_LLM_ANSWER):
    """
    Import `ask` with all external dependencies mocked.
    Returns (ask, chain_mock) so tests can inspect chain.invoke calls.
    """
    with (
        patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=FAKE_SYSTEM_PROMPT))),
                __exit__=MagicMock(return_value=False),
            )
        )),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", MagicMock()),
    ):
        import sys
        for mod in ["backend.config", "backend.prompts", "backend.llm_client"]:
            sys.modules.pop(mod, None)

        import backend.llm_client as llm_module

    # Patch chain.invoke on the live module object
    chain_mock = MagicMock(return_value=mock_answer)
    llm_module.chain = MagicMock()
    llm_module.chain.invoke = chain_mock

    # Patch the prompt builders so they return predictable strings
    llm_module.build_system_prompt = MagicMock(return_value=FAKE_SYSTEM_PROMPT)
    llm_module.build_user_prompt = MagicMock(side_effect=lambda **kwargs: kwargs["user_question"])

    return llm_module.ask, chain_mock


# ---------------------------------------------------------------------------
# Tests: ask() return structure
# ---------------------------------------------------------------------------

class TestAskReturnStructure:

    def test_returns_answer_key(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert "answer" in result

    def test_returns_updated_history_key(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert "updated_history" in result

    def test_answer_matches_mock(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert result["answer"] == FAKE_LLM_ANSWER

    def test_no_error_key_on_success(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert "error" not in result


# ---------------------------------------------------------------------------
# Tests: conversation history handling
# ---------------------------------------------------------------------------

class TestConversationHistory:

    def test_empty_history_produces_two_new_turns(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert len(result["updated_history"]) == 2

    def test_history_grows_by_two_each_turn(self):
        ask, _ = make_ask()
        r1 = ask("What is variance?")
        r2 = ask("Give an example.", conversation_history=r1["updated_history"])
        assert len(r2["updated_history"]) == 4

    def test_updated_history_last_user_role(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert result["updated_history"][-2]["role"] == "user"

    def test_updated_history_last_assistant_role(self):
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert result["updated_history"][-1]["role"] == "assistant"

    def test_updated_history_stores_raw_question(self):
        # History should store the original question, not the composed prompt
        ask, _ = make_ask()
        result = ask("What is variance?")
        assert result["updated_history"][-2]["content"] == "What is variance?"

    def test_updated_history_preserves_existing_turns(self):
        ask, _ = make_ask()
        prior_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = ask("What is variance?", conversation_history=prior_history)
        assert result["updated_history"][:2] == prior_history

    def test_none_history_treated_as_empty(self):
        ask, _ = make_ask()
        result = ask("What is variance?", conversation_history=None)
        assert len(result["updated_history"]) == 2


# ---------------------------------------------------------------------------
# Tests: chain.invoke receives correct keys
# ---------------------------------------------------------------------------

class TestChainInvokePayload:

    def test_invoke_called_once_per_ask(self):
        ask, chain_mock = make_ask()
        ask("What is variance?")
        chain_mock.assert_called_once()

    def test_invoke_receives_system_prompt_key(self):
        ask, chain_mock = make_ask()
        ask("What is variance?")
        payload = chain_mock.call_args[0][0]
        assert "system_prompt" in payload

    def test_invoke_receives_user_message_key(self):
        ask, chain_mock = make_ask()
        ask("What is variance?")
        payload = chain_mock.call_args[0][0]
        assert "user_message" in payload

    def test_invoke_receives_chat_history_key(self):
        ask, chain_mock = make_ask()
        ask("What is variance?")
        payload = chain_mock.call_args[0][0]
        assert "chat_history" in payload

    def test_system_prompt_built_with_user_profile(self):
        # build_system_prompt should be called with the profile fields
        ask, chain_mock = make_ask()
        import sys
        llm_module = sys.modules["backend.llm_client"]
        ask("What is variance?", user_level="Beginner", user_goal="pass exam", user_background="biology")
        llm_module.build_system_prompt.assert_called_once_with(
            user_level="Beginner",
            user_goal="pass exam",
            user_background="biology",
        )

    def test_user_prompt_built_with_correct_args(self):
        # build_user_prompt should receive the question and profile fields
        ask, chain_mock = make_ask()
        import sys
        llm_module = sys.modules["backend.llm_client"]
        ask("What is variance?", user_level="Beginner", user_goal="pass exam", user_background="biology")
        call_kwargs = llm_module.build_user_prompt.call_args[1]
        assert call_kwargs["user_level"] == "Beginner"
        assert call_kwargs["user_goal"] == "pass exam"
        assert call_kwargs["user_background"] == "biology"


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_returns_error_key_on_exception(self):
        ask, chain_mock = make_ask()
        chain_mock.side_effect = Exception("API timeout")
        result = ask("What is variance?")
        assert "error" in result

    def test_error_message_propagated(self):
        ask, chain_mock = make_ask()
        chain_mock.side_effect = Exception("API timeout")
        result = ask("What is variance?")
        assert "API timeout" in result["error"]

    def test_no_answer_key_on_error(self):
        ask, chain_mock = make_ask()
        chain_mock.side_effect = Exception("API timeout")
        result = ask("What is variance?")
        assert "answer" not in result


# ---------------------------------------------------------------------------
# Tests: build_rag_context (pure function, no mocking needed)
# ---------------------------------------------------------------------------

class TestBuildRagContext:

    @pytest.fixture(autouse=True)
    def import_prompts(self):
        with (
            patch("builtins.open", MagicMock(
                return_value=MagicMock(
                    __enter__=MagicMock(return_value=MagicMock(read=MagicMock(return_value=FAKE_SYSTEM_PROMPT))),
                    __exit__=MagicMock(return_value=False),
                )
            )),
        ):
            import sys
            sys.modules.pop("backend.prompts", None)
            sys.modules.pop("backend.config", None)
            import backend.prompts as prompts_module
            self.build_rag_context = prompts_module.build_rag_context

    def test_empty_list_returns_empty_string(self):
        assert self.build_rag_context([]) == ""

    def test_single_chunk_included_in_output(self):
        result = self.build_rag_context(["Chunk one content."])
        assert "Chunk one content." in result

    def test_multiple_chunks_all_included(self):
        chunks = ["First chunk.", "Second chunk.", "Third chunk."]
        result = self.build_rag_context(chunks)
        for chunk in chunks:
            assert chunk in result

    def test_chunks_separated_by_delimiter(self):
        result = self.build_rag_context(["A", "B"])
        assert "---" in result

    def test_output_contains_instruction_prefix(self):
        result = self.build_rag_context(["Some context."])
        assert result.startswith("Use the following context")