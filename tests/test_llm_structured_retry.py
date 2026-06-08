"""Tests for llm-structured-retry-py.

These tests use only the Python standard library (``unittest``) so they run
without any third-party dependencies. Run them with::

    python3 -m unittest discover -s tests
"""

import json
import unittest

from llm_structured_retry import (
    RetryAttempt,
    RetryResult,
    StructuredRetry,
    StructuredRetryExhausted,
    structured_retry,
)


def make_call_fn(responses):
    """Return a ``call_fn`` that cycles through a list of responses."""
    responses = list(responses)
    idx = [0]

    def call_fn(messages):
        r = responses[idx[0] % len(responses)]
        idx[0] += 1
        return r

    return call_fn


def json_parse_fn(output):
    """Parse JSON from the ``content`` field of an assistant message."""
    content = output.get("content", "")
    return json.loads(content)


class StructuredRetryTests(unittest.TestCase):
    def test_success_first_attempt(self):
        responses = [{"role": "assistant", "content": '{"ok": true}'}]
        retry = StructuredRetry(max_attempts=3)
        result = retry.run(
            messages=[{"role": "user", "content": "Return JSON"}],
            call_fn=make_call_fn(responses),
            parse_fn=json_parse_fn,
        )
        self.assertEqual(result.value, {"ok": True})
        self.assertEqual(result.attempts, 1)
        self.assertTrue(result.ok)
        self.assertEqual(result.history, [])

    def test_retries_on_invalid_then_succeeds(self):
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
        self.assertEqual(result.value, {"ok": True})
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.history), 1)
        self.assertEqual(result.history[0].attempt, 1)
        self.assertEqual(result.history[0].raw_output, responses[0])

    def test_exhausted_raises(self):
        responses = [{"role": "assistant", "content": "not json"}]
        retry = StructuredRetry(max_attempts=3)
        with self.assertRaises(StructuredRetryExhausted) as ctx:
            retry.run(
                messages=[{"role": "user", "content": "Return JSON"}],
                call_fn=make_call_fn(responses),
                parse_fn=json_parse_fn,
            )
        self.assertEqual(ctx.exception.attempts, 3)
        self.assertGreater(len(ctx.exception.last_error), 0)
        self.assertIn("3 attempt", str(ctx.exception))

    def test_history_tracks_attempts(self):
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
        self.assertEqual(result.attempts, 3)
        # Two failed attempts recorded before the third succeeds.
        self.assertEqual(len(result.history), 2)
        self.assertEqual([a.attempt for a in result.history], [1, 2])

    def test_on_retry_callback(self):
        seen = []

        def on_retry(attempt):
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
        self.assertEqual(seen, [1])

    def test_on_retry_called_on_every_failure_including_last(self):
        """The callback fires for the final failing attempt too."""
        seen = []
        responses = [{"role": "assistant", "content": "bad"}]
        retry = StructuredRetry(max_attempts=3, on_retry=seen.append)
        with self.assertRaises(StructuredRetryExhausted):
            retry.run(
                messages=[{"role": "user", "content": "JSON"}],
                call_fn=make_call_fn(responses),
                parse_fn=json_parse_fn,
            )
        self.assertEqual([a.attempt for a in seen], [1, 2, 3])

    def test_error_injected_in_messages(self):
        """The validation error is injected as a follow-up user message."""
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
        self.assertEqual(len(received_messages), 2)
        last_msgs = received_messages[1]
        self.assertTrue(
            any(
                "Error:" in m.get("content", "")
                for m in last_msgs
                if m["role"] == "user"
            )
        )

    def test_assistant_response_injected_before_error(self):
        """A dict response with content is echoed back before the error."""
        received = []

        def call_fn(messages):
            received.append(list(messages))
            if len(received) == 1:
                return {"role": "assistant", "content": "bad"}
            return {"role": "assistant", "content": '{"ok": true}'}

        StructuredRetry(max_attempts=2).run(
            messages=[{"role": "user", "content": "Return JSON"}],
            call_fn=call_fn,
            parse_fn=json_parse_fn,
        )
        second_call_msgs = received[1]
        # user, assistant(bad), user(error)
        self.assertEqual(len(second_call_msgs), 3)
        self.assertEqual(second_call_msgs[1], {"role": "assistant", "content": "bad"})
        self.assertEqual(second_call_msgs[2]["role"], "user")

    def test_non_dict_raw_output_still_injects_error(self):
        """When the raw output isn't a dict, only the error message is added."""
        received = []

        def call_fn(messages):
            received.append(list(messages))
            if len(received) == 1:
                return "not json"  # plain string, not a message dict
            return '{"ok": true}'

        def parse_str(output):
            return json.loads(output)

        result = StructuredRetry(max_attempts=2).run(
            messages=[{"role": "user", "content": "Return JSON"}],
            call_fn=call_fn,
            parse_fn=parse_str,
        )
        self.assertEqual(result.value, {"ok": True})
        # Original user message + injected error message only (no echoed dict).
        second_call_msgs = received[1]
        self.assertEqual(len(second_call_msgs), 2)
        self.assertEqual(second_call_msgs[1]["role"], "user")

    def test_input_messages_not_mutated(self):
        """The caller's message list must not be mutated in place."""
        original = [{"role": "user", "content": "Return JSON"}]
        snapshot = json.dumps(original)
        responses = [
            {"role": "assistant", "content": "bad"},
            {"role": "assistant", "content": '{"ok": true}'},
        ]
        StructuredRetry(max_attempts=3).run(
            messages=original,
            call_fn=make_call_fn(responses),
            parse_fn=json_parse_fn,
        )
        self.assertEqual(json.dumps(original), snapshot)
        self.assertEqual(len(original), 1)

    def test_custom_prefix_and_suffix(self):
        captured = []

        def call_fn(messages):
            captured.append(list(messages))
            if len(captured) == 1:
                return {"role": "assistant", "content": "bad"}
            return {"role": "assistant", "content": '{"ok": true}'}

        StructuredRetry(
            max_attempts=2,
            error_prefix="<<PREFIX>>",
            instruction_suffix="<<SUFFIX>>",
        ).run(
            messages=[{"role": "user", "content": "go"}],
            call_fn=call_fn,
            parse_fn=json_parse_fn,
        )
        injected = captured[1][-1]["content"]
        self.assertTrue(injected.startswith("<<PREFIX>>"))
        self.assertTrue(injected.endswith("<<SUFFIX>>"))

    def test_wrap_returns_callable(self):
        responses = [{"role": "assistant", "content": '{"x": 1}'}]
        retry = StructuredRetry(max_attempts=2)
        runner = retry.wrap(make_call_fn(responses), json_parse_fn)
        result = runner([{"role": "user", "content": "JSON"}])
        self.assertEqual(result.value, {"x": 1})

    def test_structured_retry_convenience(self):
        responses = [{"role": "assistant", "content": '{"y": 2}'}]
        result = structured_retry(
            messages=[{"role": "user", "content": "JSON"}],
            call_fn=make_call_fn(responses),
            parse_fn=json_parse_fn,
            max_attempts=3,
        )
        self.assertEqual(result.value, {"y": 2})

    def test_structured_retry_passes_on_retry_callback(self):
        seen = []
        responses = [
            {"role": "assistant", "content": "bad"},
            {"role": "assistant", "content": '{"z": 3}'},
        ]
        result = structured_retry(
            messages=[{"role": "user", "content": "JSON"}],
            call_fn=make_call_fn(responses),
            parse_fn=json_parse_fn,
            max_attempts=3,
            on_retry=seen.append,
        )
        self.assertEqual(result.value, {"z": 3})
        self.assertEqual([a.attempt for a in seen], [1])

    def test_max_attempts_one_no_retry(self):
        """With max_attempts=1 a single failure exhausts immediately."""
        calls = []

        def call_fn(messages):
            calls.append(1)
            return {"role": "assistant", "content": "bad"}

        with self.assertRaises(StructuredRetryExhausted) as ctx:
            StructuredRetry(max_attempts=1).run(
                messages=[{"role": "user", "content": "JSON"}],
                call_fn=call_fn,
                parse_fn=json_parse_fn,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(ctx.exception.attempts, 1)

    def test_invalid_max_attempts_raises(self):
        with self.assertRaises(ValueError):
            StructuredRetry(max_attempts=0)
        with self.assertRaises(ValueError):
            StructuredRetry(max_attempts=-1)


class DataclassTests(unittest.TestCase):
    def test_retry_attempt_dataclass(self):
        ra = RetryAttempt(attempt=1, raw_output={"content": "bad"}, error="json error")
        self.assertEqual(ra.attempt, 1)
        self.assertEqual(ra.error, "json error")
        self.assertEqual(ra.raw_output, {"content": "bad"})

    def test_retry_result_ok(self):
        rr = RetryResult(value=42, attempts=1)
        self.assertTrue(rr.ok)
        self.assertEqual(rr.value, 42)
        self.assertEqual(rr.history, [])

    def test_exhausted_with_no_history(self):
        """The exception's repr survives an empty last_error."""
        exc = StructuredRetryExhausted(2, "")
        self.assertEqual(exc.attempts, 2)
        self.assertEqual(exc.last_error, "")
        self.assertIn("2 attempt", str(exc))


if __name__ == "__main__":
    unittest.main()
