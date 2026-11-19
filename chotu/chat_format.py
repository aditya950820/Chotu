"""Plain-text ChatML and Hermes-style tools; no tokenizer vocabulary changes.

Messages have role/content and optional assistant tool_calls (name/arguments dicts).
Parsing returns (calls, errors). Tokenization returns None when over the limit.
This module formats calls, but never executes them.
"""
import json
import re

from chotu.config import FINAL_CONTEXT


def _validate_call(call):
    if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"]:
        raise ValueError("tool call needs a nonempty name")
    if not isinstance(call.get("arguments"), dict):
        raise ValueError("tool call arguments must be an object")
    return {"name": call["name"], "arguments": call["arguments"]}


def _turns(messages, tools=None):
    messages = [dict(message) for message in messages]
    if tools:
        instruction = ('Available tools are listed below. Use tool_call tags around '
                       'a JSON object with name and arguments; '
                       'use multiple calls for parallel requests.\n<tools>\n'
                       + json.dumps(tools, ensure_ascii=False) + '\n</tools>')
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = messages[0].get("content", "") + "\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {role}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("content must be text")
        if message.get("tool_calls"):
            if role != "assistant":
                raise ValueError("only assistants can call tools")
            for call in message["tool_calls"]:
                content += "<tool_call>" + json.dumps(_validate_call(call), ensure_ascii=False) + "</tool_call>"
        if role == "tool":
            content = "<tool_response>" + content + "</tool_response>"
        yield role, f"<|im_start|>{role}\n", content + "<|im_end|>", "\n"


def render_conversation(messages, tools=None):
    """Render complete turns; tool schemas are JSON in the first system turn."""
    return "".join(header + body + newline for _, header, body, newline in _turns(messages, tools))


def parse_tool_calls(text):
    """Keep valid calls and report invalid or unclosed blocks without raising."""
    calls, errors = [], []
    pattern = r"<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>)|$)"
    for match in re.finditer(pattern, text, re.DOTALL):
        if not match.group(0).endswith("</tool_call>"):
            errors.append("unclosed tool_call")
            continue
        try:
            calls.append(_validate_call(json.loads(match.group(1))))
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    return calls, errors


def tokenize_conversation(messages, tokenizer, tools=None, max_length=FINAL_CONTEXT):
    """Mask headers and non-assistant turns; include assistant end-of-turn tokens.

    Tokenize the rendered text once, using offsets so masks match its actual BPE
    boundaries. No truncation: return None rather than train on partial turns.
    """
    text, spans = "", []
    for role, header, body, newline in _turns(messages, tools):
        start = len(text) + len(header)
        text += header + body + newline
        if role == "assistant":
            spans.append((start, start + len(body)))
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    if len(ids) > max_length:
        return None
    labels = [token if any(start <= a < b <= end for start, end in spans) else -100
              for token, (a, b) in zip(ids, encoded["offset_mapping"])]
    return {"input_ids": ids, "labels": labels}
