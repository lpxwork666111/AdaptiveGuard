from adaptiveguard.io.serialization import append_jsonl, read_json, read_jsonl, write_json


def test_json_and_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "nested" / "data.json"
    write_json(path, {"value": "测试"})
    assert read_json(path) == {"value": "测试"}
    jsonl = tmp_path / "records.jsonl"
    append_jsonl(jsonl, {"id": 1})
    append_jsonl(jsonl, {"id": 2})
    assert read_jsonl(jsonl) == [{"id": 1}, {"id": 2}]


def test_json_serialization_uses_string_fallback_for_nested_values(tmp_path) -> None:
    path = tmp_path / "value.json"

    write_json(path, {"value": object()})

    loaded = read_json(path)
    assert isinstance(loaded["value"], str)
