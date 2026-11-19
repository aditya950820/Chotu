import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import httpx
from huggingface_hub.errors import RepositoryNotFoundError

from chotu.tool_eval import apigen_score, bfcl_score, bfcl_tools, allowed_value


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = script("export_gguf")
upload = script("upload")
lm = script("eval_lm")
tools = script("eval_tools")


def output(*calls):
    return "".join("<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls)


FUNCTIONS = [{"name": "area", "parameters": {"type": "dict", "properties": {
    "base": {"type": "integer"}, "height": {"type": "integer"}, "unit": {"type": "string"}},
    "required": ["base", "height"]}}]
GOLD = [{"area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
CALL = {"name": "area", "arguments": {"base": 10, "height": 5}}


def test_bfcl_optional_and_allowed():
    assert bfcl_score(output(CALL), GOLD, FUNCTIONS) == {"json_valid": True, "accuracy": True}
    assert bfcl_score(output({**CALL, "arguments": {**CALL["arguments"], "unit": "UN-ITS"}}), GOLD, FUNCTIONS)["accuracy"]
    mandatory_optional = [{"area": {"base": [10], "height": [5], "unit": ["units"]}}]
    assert not bfcl_score(output(CALL), mandatory_optional, FUNCTIONS)["accuracy"]
    assert not bfcl_score(output({**CALL, "arguments": {"base": 10}}), GOLD, FUNCTIONS)["accuracy"]


@pytest.mark.parametrize("args", [
    {"base": "10", "height": 5}, {"base": True, "height": 5},
    {"base": 10.0, "height": 5}, {"base": 10, "height": 5, "extra": 1},
    {"base": 11, "height": 5}, {"base": 10, "height": 5, "unit": 3},
])
def test_bfcl_wrong_types_and_values(args):
    assert not bfcl_score(output({"name": "area", "arguments": args}), GOLD, FUNCTIONS)["accuracy"]


def test_parallel_matching_and_multiplicity():
    other = {"name": "area", "arguments": {"base": 20, "height": 5}}
    gold = GOLD + [{"area": {"base": [20], "height": [5], "unit": [""]}}]
    assert bfcl_score(output(other, CALL), gold, FUNCTIONS)["accuracy"]
    assert not bfcl_score(output(CALL, CALL), gold, FUNCTIONS)["accuracy"]
    assert not bfcl_score(output(CALL), gold, FUNCTIONS)["accuracy"]
    # An ambiguous first call must not steal the only match for the second.
    gold[0]["area"]["base"] = [10, 20]
    assert bfcl_score(output(other, CALL), gold[::-1], FUNCTIONS)["accuracy"]
    assert not bfcl_score(output({"name": "wrong", "arguments": {}}), GOLD, FUNCTIONS)["accuracy"]


@pytest.mark.parametrize("value,choices,schema,expected", [
    (3, [3.0], {"type": "float"}, True),
    (3, [3], {"type": "float"}, True),
    (True, [1], {"type": "float"}, False),
    ("x", ["x"], {"type": "integer"}, True),
    ("X", ["x"], {"type": "integer"}, False),
    ([1, 2], [[1, 2]], {"type": "array", "items": {"type": "integer"}}, True),
    ([1, True], [[1, 1]], {"type": "array", "items": {"type": "integer"}}, False),
    ({"x": "IN-CHES"}, [{"x": ["inches"], "optional": [""]}], {"type": "dict"}, True),
    ({"y": 1}, [{"x": [1]}], {"type": "dict"}, False),
    ([5], [[{"x": [1]}]], {"type": "array", "items": {"type": "dict"}}, False),
])
def test_allowed_values(value, choices, schema, expected):
    assert allowed_value(value, choices, schema) is expected


def test_bfcl_tools_schema():
    converted = bfcl_tools(FUNCTIONS)
    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["parameters"]["type"] == "object"
    assert FUNCTIONS[0]["parameters"]["type"] == "dict"
    special = [{"name": "f", "parameters": {"type": "dict", "properties": {"type": {"type": "float"}}}}]
    assert bfcl_tools(special)[0]["function"]["parameters"]["properties"]["type"]["type"] == "number"


@pytest.mark.parametrize("text", ["", "hello", '<tool_call>{oops}</tool_call>',
                                  '<tool_call>{"name":"f","arguments":{"x":NaN}}</tool_call>',
                                  'prefix <tool_call>{"name":"area","arguments":{}}</tool_call>'])
def test_invalid_tool_outputs(text):
    assert not bfcl_score(text, GOLD, FUNCTIONS)["accuracy"]
    assert not bfcl_score(text, GOLD, FUNCTIONS)["json_valid"]
    assert not apigen_score(text, [CALL])["json_valid"]


def test_apigen_scoring():
    other = {"name": "f", "arguments": {"x": [1, 2]}}
    assert apigen_score(output(other, CALL), [CALL, other]) == {"json_valid": True, "exact_match": True, "name_accuracy": True}
    wrong = {**CALL, "arguments": {"base": 3}}
    assert apigen_score(output(wrong), [CALL]) == {"json_valid": True, "exact_match": False, "name_accuracy": True}
    assert not apigen_score(output(CALL, CALL), [CALL])["exact_match"]
    assert apigen_score("No available tool.", [])["exact_match"]
    assert apigen_score("No available tool.", [])["json_valid"] is None
    assert not apigen_score("", [])["exact_match"]


def test_modelfile():
    text = export.modelfile()
    for fragment in ('FROM ./chotu-q8_0.gguf', 'PARAMETER num_ctx 8192', 'PARAMETER stop "<|im_end|>"',
                     '.System', '.Tools', '.Messages', '.ToolCalls', '<tool_call>', '<tool_response>', '<|im_start|>assistant'):
        assert fragment in text


@pytest.fixture
def artifacts(tmp_path):
    model = tmp_path / "best"
    model.mkdir()
    (model / "config.json").write_text('{}')
    (model / "model.safetensors").write_bytes(b'fixture')
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b'fixture')
    return [model, gguf]


def test_private_upload_only(artifacts):
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(private=True)
    upload.upload("owner/private", artifacts, api)
    api.create_repo.assert_not_called()
    kwargs = api.create_commit.call_args.kwargs
    names = {op.path_in_repo for op in kwargs["operations"]}
    assert names == {"config.json", "model.safetensors", "model.gguf", "README.md"}
    api.repo_info.assert_called_once_with(repo_id="owner/private", repo_type="model")


@pytest.mark.parametrize("private", [False, None, "true"])
def test_upload_refuses_public_or_unknown(artifacts, private):
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(private=private)
    with pytest.raises(ValueError, match="must be private"):
        upload.upload("owner/public", artifacts, api)
    api.create_commit.assert_not_called()
    api.create_repo.assert_not_called()


def test_create_private_repo_and_recheck(artifacts):
    api = MagicMock()
    missing = RepositoryNotFoundError("missing", response=httpx.Response(404, request=httpx.Request("GET", "https://example.invalid")))
    api.repo_info.side_effect = [missing, SimpleNamespace(private=True)]
    upload.upload("owner/new", artifacts, api)
    api.create_repo.assert_called_once_with(repo_id="owner/new", repo_type="model", private=True, exist_ok=True)
    assert api.repo_info.call_count == 2
    api = MagicMock()
    api.repo_info.side_effect = [missing, SimpleNamespace(private=False)]
    with pytest.raises(ValueError):
        upload.upload("owner/race", artifacts, api)
    api.create_commit.assert_not_called()


def test_upload_input_errors(artifacts, tmp_path):
    api = MagicMock()
    for paths in ([], [tmp_path / "missing"], [artifacts[1], artifacts[1]]):
        with pytest.raises(ValueError):
            upload.upload("owner/private", paths, api)
    (artifacts[0] / ".env").write_text("fixture only")
    with pytest.raises(ValueError, match="hidden/credential"):
        upload.upload("owner/private", artifacts, api)
    api.repo_info.assert_not_called()


def test_lm_cache_and_settings(tmp_path, monkeypatch):
    import sys
    calls = []
    monkeypatch.setitem(sys.modules, "lm_eval.models.huggingface", SimpleNamespace(HFLM=lambda **kw: calls.append(kw)))
    evaluate = MagicMock(return_value={"results": {"hellaswag": {"acc,none": .1, "acc_norm,none": .2}}})
    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=evaluate))
    monkeypatch.setattr(lm, "model_identity", lambda model: {"revision": "pinned"})
    settings = dict(device="cpu", batch_size=1, tasks=["hellaswag"], limit=2)
    assert lm.evaluate_one("gpt2", settings, tmp_path) == lm.evaluate_one("gpt2", settings, tmp_path)
    assert len(calls) == evaluate.call_count == 1
    assert calls[0]["max_length"] == 1024 and calls[0]["revision"] == "pinned"
    assert evaluate.call_args.kwargs["num_fewshot"] == 0
    assert evaluate.call_args.kwargs["apply_chat_template"] is False
    lm.evaluate_one("gpt2", {**settings, "limit": 3}, tmp_path)
    assert len(calls) == 2


def test_lm_verdict_and_missing_metric():
    tasks = ["hellaswag", "arc_easy", "winogrande"]
    def report(name, score):
        return {"model": name, "results": {t: {"acc,none": score, **({"acc_norm,none": score} if t != "winogrande" else {})} for t in tasks}}
    reports = [report("target", .6), report("gpt2", .5)]
    table, gate = lm.comparison(reports, tasks, False)
    assert "N/A" in table and gate["status"] == "pass"
    assert lm.comparison(reports, tasks, True)[1]["status"] == "smoke_only"
    assert lm.comparison(reports[:1], tasks, False)[1]["status"] == "incomplete"
    assert lm.comparison([report("target", .5), reports[1]], tasks, False)[1]["status"] == "fail"


def test_summary_empty_and_failures():
    assert tools.summarize([]) == {"count": 0, "failures": []}
    rows = [{"score": {"json_valid": True, "accuracy": False}, "output": "x"}] * 10
    summary = tools.summarize(rows)
    assert summary["accuracy"] == 0 and summary["json_valid"] == 1 and len(summary["failures"]) == 5
    assert tools.summarize([{"score": apigen_score("No tool.", [])}])["json_count"] == 0
