import json

import pytest
from transformers import AutoTokenizer

from chotu.chat_data import convert_chat
from chotu.chat_format import parse_tool_calls, render_conversation, tokenize_conversation
from chotu.config import TOKENIZER_ID
from chotu.tool_eval import apigen_score


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


@pytest.mark.parametrize("value", ["</tool_call>", "<tool_call>", 'both <tool_call> and </tool_call> with "quotes" and \\'])
def test_literal_tags_in_arguments(value):
    calls = [{"name": "echo", "arguments": {"nested": [{"text": value}]}}]
    text = render_conversation([{"role": "assistant", "tool_calls": calls}])
    assert parse_tool_calls(text) == (calls, [])
    tags = text.split("\n", 1)[1].removesuffix("<|im_end|>\n")
    assert parse_tool_calls(tags, allow_text=False) == (calls, [])
    assert apigen_score(tags, calls) == {"json_valid": True, "exact_match": True, "name_accuracy": True}


def test_apigen_list_conversion_with_literal_tags():
    calls = [{"name": "echo", "arguments": {"text": "<tool_call>literal</tool_call>"}}]
    tools = [{"name": "echo", "parameters": {"type": "object"}}]
    row = {"messages": [
        {"role": "system", "content": "<tools>" + json.dumps(tools) + "</tools>"},
        {"role": "user", "content": "Echo these tags."},
        {"role": "assistant", "content": "<tool_call>" + json.dumps(calls) + "</tool_call>"},
    ]}
    messages, schemas, normalized = convert_chat(row, "apigen")
    assert normalized == calls
    assert parse_tool_calls(render_conversation(messages, schemas)) == (calls, [])


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_nested_json_rejected(constant):
    bad = '<tool_call>{"name":"bad","arguments":{"nested":[{"value":' + constant + '}]}}</tool_call>'
    valid = '<tool_call>{"name":"ok","arguments":{}}</tool_call>'
    calls, errors = parse_tool_calls(bad + valid)
    assert calls == [{"name": "ok", "arguments": {}}]
    assert len(errors) == 1 and "non-finite" in errors[0]
    value = {"nested": [{"value": float(constant)}]}
    with pytest.raises(ValueError):
        render_conversation([{"role": "assistant", "tool_calls": [{"name": "bad", "arguments": value}]}])
    with pytest.raises(ValueError):
        render_conversation([], tools=[{"name": "bad", "parameters": value}])


def test_deep_json_recovers_later_call():
    depth = 100_000
    bad = '<tool_call>{"name":"bad","arguments":{"nested":' + '[' * depth + '0' + ']' * depth + '}}</tool_call>'
    calls, errors = parse_tool_calls(bad + '<tool_call>{"name":"ok","arguments":{}}</tool_call>')
    assert calls == [{"name": "ok", "arguments": {}}]
    assert len(errors) == 1 and "recursion" in errors[0].lower()


def test_missing_closer_and_strict_text():
    valid = '<tool_call>{"name":"ok","arguments":{}} \n\t</tool_call>'
    assert parse_tool_calls('<tool_call>{"name":"unclosed","arguments":{}}' + valid) == (
        [{"name": "ok", "arguments": {}}], ["unclosed tool_call"])
    assert parse_tool_calls("prose " + valid, allow_text=False)[1] == ["text outside tool_call"]
    assert parse_tool_calls(valid + " prose", allow_text=False)[1] == ["text outside tool_call"]
    assert parse_tool_calls("", allow_text=False) == ([], [])


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
