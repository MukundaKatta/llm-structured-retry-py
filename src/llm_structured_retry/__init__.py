"""llm-structured-retry-py — retry LLM calls by injecting validation errors as follow-up messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


class StructuredRetryExhausted(Exception):
    """Raised when all structured-retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: str) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Structured retry failed after {attempts} attempt(s): {last_error}"
        )


@dataclass
class RetryAttempt:
    """Record of a single structured-retry attempt."""

    attempt: int
    raw_output: Any
    error: str


@dataclass
class RetryResult:
    """Outcome of a structured retry run."""

    value: Any
    attempts: int
    history: list[RetryAttempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return True  # RetryResult is only returned on success


class StructuredRetry:
    """
    Retry LLM calls by injecting validation errors into the conversation.

    When the LLM returns output that fails a validator, the error is appended
    as a user message and the call is retried. This is more effective than
    silent retries because the model sees what went wrong.

    Example::

        import json

        def call_llm(messages):
            # your actual LLM call here
            return {"role": "assistant", "content": '{"name": "Alice"}'}

        def parse_json(output):
            content = output["content"]
            parsed = json.loads(content)
            if "name" not in parsed:
                raise ValueError("Missing 'name' field")
            return parsed

        retry = StructuredRetry(max_attempts=3)
        result = retry.run(
            messages=[{"role": "user", "content": "Return JSON with a name field."}],
            call_fn=call_llm,
            parse_fn=parse_json,
        )
        print(result.value)   # {"name": "Alice"}
        print(result.attempts)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        error_prefix: str = "Your previous response was invalid. Error: ",
        instruction_suffix: str = " Please try again and fix the issue.",
        on_retry: Callable[[RetryAttempt], None] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self.error_prefix = error_prefix
        self.instruction_suffix = instruction_suffix
        self.on_retry = on_retry

    def run(
        self,
        messages: list[dict],
        call_fn: Callable[[list[dict]], Any],
        parse_fn: Callable[[Any], Any],
    ) -> RetryResult:
        """
        Run the structured retry loop.

        Args:
            messages: Initial message list for the LLM.
            call_fn: Callable that takes messages and returns LLM output.
            parse_fn: Callable that validates/parses LLM output.
                      Should raise ValueError (or any Exception) on failure.

        Returns:
            RetryResult with the parsed value and attempt count.

        Raises:
            StructuredRetryExhausted: If all attempts fail.
        """
        current_messages = list(messages)
        history: list[RetryAttempt] = []

        for attempt in range(1, self.max_attempts + 1):
            raw = call_fn(current_messages)
            try:
                value = parse_fn(raw)
                return RetryResult(value=value, attempts=attempt, history=history)
            except Exception as exc:
                error_str = str(exc)
                rec = RetryAttempt(attempt=attempt, raw_output=raw, error=error_str)
                history.append(rec)
                if self.on_retry:
                    self.on_retry(rec)
                if attempt < self.max_attempts:
                    # Inject the assistant response + error feedback
                    if isinstance(raw, dict) and "content" in raw:
                        current_messages = current_messages + [raw]
                    error_msg = {
                        "role": "user",
                        "content": (
                            f"{self.error_prefix}{error_str}{self.instruction_suffix}"
                        ),
                    }
                    current_messages = current_messages + [error_msg]

        raise StructuredRetryExhausted(
            self.max_attempts, history[-1].error if history else ""
        )

    def wrap(
        self,
        call_fn: Callable[[list[dict]], Any],
        parse_fn: Callable[[Any], Any],
    ) -> Callable[[list[dict]], RetryResult]:
        """Return a callable that runs the structured retry loop."""

        def runner(messages: list[dict]) -> RetryResult:
            return self.run(messages, call_fn, parse_fn)

        return runner


def structured_retry(
    messages: list[dict],
    call_fn: Callable[[list[dict]], Any],
    parse_fn: Callable[[Any], Any],
    max_attempts: int = 3,
    error_prefix: str = "Your previous response was invalid. Error: ",
    instruction_suffix: str = " Please try again and fix the issue.",
    on_retry: Callable[[RetryAttempt], None] | None = None,
) -> RetryResult:
    """Convenience function for one-shot structured retry."""
    return StructuredRetry(
        max_attempts=max_attempts,
        error_prefix=error_prefix,
        instruction_suffix=instruction_suffix,
        on_retry=on_retry,
    ).run(messages, call_fn, parse_fn)


__all__ = [
    "StructuredRetry",
    "StructuredRetryExhausted",
    "RetryAttempt",
    "RetryResult",
    "structured_retry",
]
