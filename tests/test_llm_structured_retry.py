"""Tests for llm-structured-retry-py."""
import json
import pytest
from llm_structured_retry import (
    StructuredRetry, StructuredRetryExhausted, RetryAttempt, RetryResult, structured_retry
)


def make_call_fn(responses):
    """Returns a call_fn that cycles through a list of responses."""
    responses = list(responses)
    idx = [0]
    def call_fn(messages):
        r = responses[idx[0] % len(responses)]
        idx[0] += 1
        return r
    return call_fn


def json_parse_fn(output):
    """Parse JSON from the 'content' field of an assistant message."""
    content = output.get("content", "")
    return json.loads(content)


def test_success_first_attempt():
    responses = [{"role": "assistant", "content": '{"ok": true}'}]
    retry = StructuredRetry(max_attempts=3)
    result = retry.run(
        messages=[{"role": "user", "content": "Return JSON"}],
        call_fn=make_call_fn(responses),
        parse_fn=json_parse_fn,
    )
    assert result.value == {"ok": True}
    assert result.attempts == 1
    assert result.ok is True


def test_retries_on_invalid_then_succeeds():
    responses = [
        {"role": "assistant", "content": "not json"},
        {"role": "assistant", "content": '{"ok": true}'},
    ]
    retry = StructuredRetry(max_attempts=3)
    result = retry.run(
        messages=[{"role": "user", "content": "Return JSON"}],
        call_fn=make_call_fn(responses),
        parse_fn=json_parse_fn,
    )
    assert result.value == {"ok": True}
    assert result.attempts == 2
    assert len(result.history) == 1


def test_exhausted_raises():
    responses = [{"role": "assistant", "content": "not json"}]
    retry = StructuredRetry(max_attempts=3)
    with pytest.raises(StructuredRetryExhausted) as exc_info:
        retry.run(
            messages=[{"role": "user", "content": "Return JSON"}],
            call_fn=make_call_fn(responses),
            parse_fn=json_parse_fn,
        )
    assert exc_info.value.attempts == 3
    assert len(exc_info.value.last_error) > 0


def test_history_tracks_attempts():
    responses = [
        {"role": "assistant", "content": "bad"},
        {"role": "assistant", "content": "still bad"},
        {"role": "assistant", "content": '{"x": 1}'},
    ]
    retry = StructuredRetry(max_attempts=3)
    result = retry.run(
        messages=[{"role": "user", "content": "JSON"}],
        call_fn=make_call_fn(responses),
        parse_fn=json_parse_fn,
    )
    assert result.attempts == 3
    assert len(result.history) == 2  # 2 failed attempts recorded


def test_on_retry_callback():
    seen = []
    def on_retry(attempt: RetryAttempt):
        seen.append(attempt.attempt)

    responses = [
        {"role": "assistant", "content": "bad"},
        {"role": "assistant", "content": '{"x": 1}'},
    ]
    retry = StructuredRetry(max_attempts=3, on_retry=on_retry)
    retry.run(
        messages=[{"role": "user", "content": "JSON"}],
        call_fn=make_call_fn(responses),
        parse_fn=json_parse_fn,
    )
    assert seen == [1]


def test_error_injected_in_messages():
    """Verify the error is injected as a user message."""
    received_messages = []

    def call_fn(messages):
        received_messages.append(list(messages))
        if len(received_messages) == 1:
            return {"role": "assistant", "content": "bad"}
        return {"role": "assistant", "content": '{"ok": true}'}

    retry = StructuredRetry(max_attempts=3, error_prefix="Error: ")
    retry.run(
        messages=[{"role": "user", "content": "Return JSON"}],
        call_fn=call_fn,
        parse_fn=json_parse_fn,
    )
    # Second call should have the error message
    assert len(received_messages) == 2
    last_msgs = received_messages[1]
    assert any("Error:" in m.get("content", "") for m in last_msgs if m["role"] == "user")


def test_wrap_returns_callable():
    responses = [{"role": "assistant", "content": '{"x": 1}'}]
    retry = StructuredRetry(max_attempts=2)
    runner = retry.wrap(make_call_fn(responses), json_parse_fn)
    result = runner([{"role": "user", "content": "JSON"}])
    assert result.value == {"x": 1}


def test_structured_retry_convenience():
    responses = [{"role": "assistant", "content": '{"y": 2}'}]
    result = structured_retry(
        messages=[{"role": "user", "content": "JSON"}],
        call_fn=make_call_fn(responses),
        parse_fn=json_parse_fn,
        max_attempts=3,
    )
    assert result.value == {"y": 2}


def test_retry_attempt_dataclass():
    ra = RetryAttempt(attempt=1, raw_output={"content": "bad"}, error="json error")
    assert ra.attempt == 1
    assert ra.error == "json error"


def test_retry_result_ok():
    rr = RetryResult(value=42, attempts=1)
    assert rr.ok is True
