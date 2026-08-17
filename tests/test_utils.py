from ollama_runner import classify_result, ns_to_s


def test_ns_to_s():
    assert ns_to_s(1_500_000_000) == 1.5
    assert ns_to_s(None) is None


def test_status_completed():
    assert classify_result("answer", "stop", None) == "completed"


def test_status_truncated():
    assert classify_result("partial", "length", None) == "truncated"


def test_status_empty_final():
    assert classify_result("", "stop", None) == "empty_final"
    assert classify_result("   ", "stop", None) == "empty_final"


def test_status_error_precedence():
    assert classify_result("answer", "stop", "boom") == "error"
