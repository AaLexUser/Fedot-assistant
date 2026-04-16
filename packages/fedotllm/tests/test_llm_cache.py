import importlib
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf


def _build_fake_openai(reply_text, calls):
    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=reply_text))],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

    return FakeOpenAI


def _make_assistant(monkeypatch, tmp_path, calls, **config_overrides):
    monkeypatch.setenv("FEDOTLLM_LLM_API_KEY", "test-key")
    llm_module = importlib.import_module("fedotllm.llm.llm")
    monkeypatch.setattr(llm_module, "get_llm_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        llm_module,
        "OpenAI",
        _build_fake_openai("cached reply", calls),
    )
    config = OmegaConf.create(
        {
            "model": "gpt-test",
            "base_url": "https://example.test/v1",
            "temperature": 0,
            "max_tokens": 128,
            **config_overrides,
        }
    )
    return llm_module.AssistantChatOpenAI(config)


def test_invoke_uses_disk_cache_for_identical_requests(tmp_path, monkeypatch):
    calls = []
    messages = [{"role": "user", "content": "Hello"}]

    first_assistant = _make_assistant(monkeypatch, tmp_path, calls)
    assert first_assistant.invoke(messages) == "cached reply"
    assert len(calls) == 1
    assert "extra_body" not in calls[0]

    second_assistant = _make_assistant(monkeypatch, tmp_path, calls)
    assert second_assistant.invoke(messages) == "cached reply"
    assert len(calls) == 1

    cache_files = list(tmp_path.glob("*.txt"))
    assert len(cache_files) == 1
    assert cache_files[0].read_text(encoding="utf-8") == "cached reply"
    assert second_assistant.history_[0]["input_tokens"] == 0
    assert second_assistant.history_[0]["output_tokens"] == 0


def test_invoke_cache_key_includes_requested_fields(tmp_path, monkeypatch):
    messages = [{"role": "user", "content": "Hello"}]

    for field_name, first_value, second_value in (
        ("model", "gpt-test", "gpt-test-2"),
        ("base_url", "https://example.test/v1", "https://other.test/v1"),
        ("temperature", 0, 0.2),
        ("max_tokens", 128, 256),
        (
            "extra_body",
            {"provider": {"quantizations": ["fp8", "fp16"]}},
            {"provider": {"quantizations": ["bf16", "fp32"]}},
        ),
    ):
        calls = []
        first_assistant = _make_assistant(
            monkeypatch,
            tmp_path / field_name,
            calls,
            **{field_name: first_value},
        )
        second_assistant = _make_assistant(
            monkeypatch,
            tmp_path / field_name,
            calls,
            **{field_name: second_value},
        )

        assert first_assistant.invoke(messages) == "cached reply"
        assert second_assistant.invoke(messages) == "cached reply"
        assert len(calls) == 2


def test_invoke_passes_extra_body_to_openai(tmp_path, monkeypatch):
    calls = []
    extra_body = {"provider": {"quantizations": ["fp8", "fp16", "bf16", "fp32"]}}

    assistant = _make_assistant(
        monkeypatch,
        tmp_path,
        calls,
        extra_body=extra_body,
    )

    assert assistant.invoke([{"role": "user", "content": "Hello"}]) == "cached reply"
    assert calls[0]["extra_body"] == extra_body
    assert assistant.describe()["extra_body"] == extra_body


def test_default_config_shares_extra_body_with_caafe():
    config = importlib.import_module("fedotllm.utils.configs").load_config()
    extra_body = {"provider": {"quantizations": ["fp8", "fp16", "bf16", "fp32"]}}
    config.llm.extra_body = extra_body

    assert config.feature_transformers.models.CAAFE.extra_body == extra_body


def test_get_feature_transformers_config_resolves_extra_body():
    configs_module = importlib.import_module("fedotllm.utils.configs")
    config = configs_module.load_config()
    extra_body = {"provider": {"quantizations": ["fp8", "fp16", "bf16", "fp32"]}}
    config.llm.extra_body = extra_body

    [caafe_config] = configs_module.get_feature_transformers_config(config)

    assert caafe_config["extra_body"] == extra_body


def test_caafe_query_passes_extra_body(monkeypatch):
    caafe_module = importlib.import_module(
        "fedotllm.transformer.feature_transformers.caafe"
    )
    calls = []
    extra_body = OmegaConf.create(
        {"provider": {"quantizations": ["fp8", "fp16", "bf16", "fp32"]}}
    )

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="```python\npass\n```"))
            ]
        )

    monkeypatch.setitem(
        sys.modules, "litellm", SimpleNamespace(completion=fake_completion)
    )

    client = SimpleNamespace(
        completion_params={"model": "test-model", "max_completion_tokens": 123}
    )
    messages = [{"role": "user", "content": "Hello"}]

    response = caafe_module._caafe_litellm_client.LiteLLMClient.query(
        client,
        messages,
        extra_body=extra_body,
        max_completion_tokens=456,
    )

    assert response == "```python\npass\n```"
    assert calls[0]["extra_body"] == {
        "provider": {"quantizations": ["fp8", "fp16", "bf16", "fp32"]}
    }
    assert "max_completion_tokens" not in calls[0]
