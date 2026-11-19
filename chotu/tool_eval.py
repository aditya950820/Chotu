"""Re-implementation of BFCL's AST checker for Python simple/multiple/parallel.

Not the official harness: we parse our JSON tool tags, not Python AST output.
Uses pinned official possible_answer values, optional-argument conventions,
string normalization, type checks and one-to-one unordered call matching.
Does not cover Java/JavaScript, execution, live or agentic/multi-turn BFCL.
"""
from collections import Counter
import json
import math
import re

from chotu.chat_format import parse_tool_calls

BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
BFCL_FILES = {"simple": "simple_python", "multiple": "multiple", "parallel": "parallel"}


def parsed_output(text):
    calls, errors = parse_tool_calls(text)
    remainder = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    # No calls is not a successful JSON generation, even if the parser has no errors.
    valid = bool(calls) and not errors and not remainder
    try:
        json.dumps(calls, allow_nan=False)
    except ValueError:
        valid = False
    return calls, valid


def apigen_score(text, gold_calls):
    calls, valid = parsed_output(text)
    canonical = lambda call: json.dumps(call, sort_keys=True, ensure_ascii=False, allow_nan=False)
    exact = valid and Counter(map(canonical, calls)) == Counter(map(canonical, gold_calls))
    names = Counter(c["name"] for c in calls) == Counter(c["name"] for c in gold_calls)
    if not gold_calls:
        # A refusal has no JSON to validate, but can correctly predict no calls.
        _, errors = parse_tool_calls(text)
        no_call = bool(text.strip()) and not calls and not errors and "<tool_call" not in text
        return {"json_valid": None, "exact_match": no_call, "name_accuracy": no_call}
    return {"json_valid": valid, "exact_match": bool(exact), "name_accuracy": bool(valid and names)}


def normalize_string(value):
    return re.sub(r"[ ,./\-_*^]", "", value).lower().replace("'", '"')


def equal_value(value, expected):
    if type(value) is str and type(expected) is str:
        return normalize_string(value) == normalize_string(expected)
    if type(value) is list and type(expected) is list:
        return len(value) == len(expected) and all(equal_value(v, e) for v, e in zip(value, expected))
    if type(value) is bool or type(expected) is bool:
        return type(value) is type(expected) and value == expected
    return value == expected


def dict_value(value, expected):
    if not isinstance(value, dict) or not isinstance(expected, dict) or set(value) - set(expected):
        return False
    return all((key not in value and "" in choices) or
               (key in value and any(equal_value(value[key], e) for e in choices))
               for key, choices in expected.items())


def allowed_value(value, choices, schema):
    mapping = {"string": str, "integer": int, "float": float, "number": float,
               "boolean": bool, "array": list, "tuple": list, "dict": dict, "object": dict, "any": str}
    expected_type = mapping.get(schema.get("type"))
    if expected_type is None:
        raise ValueError(f"unsupported BFCL type: {schema.get('type')}")
    if expected_type is float and type(value) is int:
        value = float(value)
    # BFCL permits symbolic variable references when possible_answer uses strings.
    answer_type = next((type(e) for e in choices if e != ""), None)
    variable = answer_type is not None and answer_type is not expected_type
    if type(value) is not expected_type and not (variable and type(value) is answer_type):
        return False
    if type(value) is float and not math.isfinite(value):
        return False
    if variable:
        return any(value == e and (type(value) is not bool or type(e) is bool) for e in choices)
    if expected_type is dict:
        return any(dict_value(value, e) for e in choices)
    if expected_type is list:
        nested = schema.get("items", {})
        if nested.get("type") in {"dict", "object"}:
            return any(type(e) is list and len(value) == len(e) and
                       all(dict_value(v, a) for v, a in zip(value, e)) for e in choices)
        # Nested values still have to obey the documented element type.
        if nested and any(not allowed_value(v, [x for e in choices if isinstance(e, list) for x in e], nested) for v in value):
            return False
    return any(equal_value(value, e) for e in choices)


def bfcl_call_matches(call, gold, functions):
    if len(gold) != 1:
        raise ValueError("expected one BFCL function per ground-truth entry")
    name, answers = next(iter(gold.items()))
    if call["name"] != name:
        return False
    description = next((f for f in functions if f["name"] == name), None)
    if description is None:
        raise ValueError(f"missing BFCL function description: {name}")
    params, values = description["parameters"], call["arguments"]
    if set(values) - set(answers) or set(values) - set(params["properties"]):
        return False
    if any(key not in values for key in params.get("required", [])):
        return False
    return all((key not in values and "" in choices) or
               (key in values and allowed_value(values[key], choices, params["properties"][key]))
               for key, choices in answers.items())


def bfcl_score(text, gold_calls, functions):
    calls, valid = parsed_output(text)
    # Backtracking avoids greedy mismatches when the same function is called twice.
    def match(index, remaining):
        if index == len(calls):
            return not remaining
        return any(bfcl_call_matches(calls[index], gold_calls[j], functions) and
                   match(index + 1, remaining - {j}) for j in remaining)
    correct = valid and len(calls) == len(gold_calls) and match(0, set(range(len(gold_calls))))
    return {"json_valid": valid, "accuracy": bool(correct)}


def bfcl_tools(functions):
    def schema(value):
        if isinstance(value, list):
            return [schema(v) for v in value]
        if not isinstance(value, dict):
            return value
        result = {k: schema(v) for k, v in value.items()}
        if isinstance(result.get("type"), str):
            result["type"] = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}.get(result["type"], result["type"])
        return result
    return [{"type": "function", "function": schema(f)} for f in functions]
