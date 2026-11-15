from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from google.genai import errors, types

from agent.llm import GeminiClient, LLMError


def client_with_generate_content(side_effect):
    client = object.__new__(GeminiClient)
    generate_content = Mock(side_effect=side_effect)
    client._client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content))
    client._model = "test-model"
    return client, generate_content


def response_with_text(text):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def test_gemini_retries_transient_server_error(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("agent.llm.time.sleep", sleep)
    client, generate_content = client_with_generate_content([
        errors.ServerError(502, {}),
        response_with_text("recovered"),
    ])

    turn = client.generate("system", [], [])

    assert turn.text == "recovered"
    assert generate_content.call_count == 2
    sleep.assert_called_once_with(0.5)


def test_gemini_does_not_retry_non_transient_client_error(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("agent.llm.time.sleep", sleep)
    client, generate_content = client_with_generate_content(errors.ClientError(400, {}))

    with pytest.raises(LLMError, match="Gemini request failed: ClientError"):
        client.generate("system", [], [])

    generate_content.assert_called_once()
    sleep.assert_not_called()


def test_gemini_retries_are_bounded(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("agent.llm.time.sleep", sleep)
    client, generate_content = client_with_generate_content([
        errors.ServerError(502, {}),
        errors.ServerError(503, {}),
        errors.ServerError(502, {}),
    ])

    with pytest.raises(LLMError, match="Gemini request failed: ServerError"):
        client.generate("system", [], [])

    assert generate_content.call_count == 3
    assert sleep.call_args_list == [call(0.5), call(1.0)]
