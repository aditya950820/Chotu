import json

import pytest
from transformers import AutoTokenizer

from chotu.chat_format import parse_tool_calls, render_conversation, tokenize_conversation
from chotu.config import TOKENIZER_ID


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


CALLS = [{"name": "weather", "arguments": {"city": city}} for city in ["Paris", "Delhi"]]
MESSAGES = [
    {"role": "system", "content": "Be helpful."},
    {"role": "user", "content": "Weather in Paris and Delhi?"},
    {"role": "assistant", "tool_calls": CALLS},
    {"role": "tool", "content": '{"city":"Paris","temperature":12}'},
    {"role": "tool", "content": '{"city":"Delhi","temperature":28}'},
    {"role": "assistant", "content": "Paris: 12°C. Delhi: 28°C."},
]


def test_render_parse_parallel():
    tools = [{"name": "weather", "parameters": {"type": "object"}}]
    text = render_conversation(MESSAGES, tools)
    assert text.startswith("<|im_start|>system\nBe helpful.\n")
    assert "<tools>\n" + json.dumps(tools) + "\n</tools>" in text
    assert '<|im_start|>tool\n<tool_response>{"city":"Paris","temperature":12}</tool_response><|im_end|>\n' in text
    assert parse_tool_calls(text) == (CALLS, [])
    assert parse_tool_calls(' <tool_call> \n {"name":"f", "arguments":{}} \n </tool_call> ') == ([{"name": "f", "arguments": {}}], [])
    assert render_conversation([{"role": "user", "content": "Hi"}], tools).startswith("<|im_start|>system\n")
    assert MESSAGES[0]["content"] == "Be helpful."


@pytest.mark.parametrize("bad", ["{oops", '[]', '{"name":"f","arguments":[]}', '{"arguments":{}}'])
def test_malformed_call(bad):
    valid = '<tool_call>{"name":"ok","arguments":{}}</tool_call>'
    calls, errors = parse_tool_calls("<tool_call>" + bad + "</tool_call>" + valid)
    assert calls == [{"name": "ok", "arguments": {}}]
    assert len(errors) == 1


def test_unclosed_and_empty():
    assert parse_tool_calls("No tools") == ([], [])
    assert parse_tool_calls("<tool_call>{}")[1] == ["unclosed tool_call"]
    assert render_conversation([]) == ""


def test_assistant_mask(tokenizer):
    text = render_conversation(MESSAGES)
    result = tokenize_conversation(MESSAGES, tokenizer)
    assert result["input_ids"] == tokenizer.encode(text, add_special_tokens=False)
    supervised = tokenizer.decode([label for label in result["labels"] if label != -100])
    expected = ''.join('<tool_call>' + json.dumps(call) + '</tool_call>' for call in CALLS)
    assert supervised == expected + '<|im_end|>Paris: 12°C. Delhi: 28°C.<|im_end|>'
    assert len(result["labels"]) == len(result["input_ids"])
    assert tokenize_conversation([], tokenizer) == {"input_ids": [], "labels": []}
    user = tokenize_conversation([{"role": "user", "content": "Hi"}], tokenizer)
    assert set(user["labels"]) == {-100}
    empty = tokenize_conversation([{"role": "assistant", "content": ""}], tokenizer)
    assert tokenizer.decode([x for x in empty["labels"] if x != -100]) == '<|im_end|>'


def test_over_length(tokenizer):
    messages = [{"role": "user", "content": " hello" * 8192}]
    assert tokenize_conversation(messages, tokenizer) is None
    size = len(tokenize_conversation(MESSAGES, tokenizer)["input_ids"])
    assert tokenize_conversation(MESSAGES, tokenizer, max_length=size) is not None
    assert tokenize_conversation(MESSAGES, tokenizer, max_length=size - 1) is None


def test_invalid_messages():
    with pytest.raises(ValueError, match="unsupported role"):
        render_conversation([{"role": "invalid"}])
    with pytest.raises(ValueError, match="only assistants"):
        render_conversation([{"role": "user", "tool_calls": CALLS}])
    with pytest.raises(ValueError, match="content must be text"):
        render_conversation([{"role": "tool", "content": {}}])
