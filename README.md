# llm-structured-retry-py

Retry LLM calls by injecting validation errors back into the conversation. More effective than silent retries because the model sees what went wrong.

## Why

When you ask an LLM for structured output (JSON, a specific schema, a value that
must pass a check) it sometimes returns something invalid. A plain retry just
runs the same prompt again and hopes for a different roll of the dice. This
library instead appends the assistant's failing response **and** the validation
error as follow-up messages, so the next attempt is grounded in what actually
went wrong:

```
user:      Return JSON with a 'name' field.
assistant: {"age": 30}                         # ← failed validation
user:      Your previous response was invalid.  # ← injected automatically
           Error: Missing required field: name. Please try again and fix the issue.
assistant: {"name": "Alice"}                    # ← now it has the feedback
```

It is provider-agnostic: you supply the function that calls your LLM and the
function that validates the output. There are no runtime dependencies.

## Install

```bash
pip install llm-structured-retry-py
```

## Usage

```python
import json
from llm_structured_retry import StructuredRetry, structured_retry

def call_llm(messages):
    # your actual LLM client here
    return client.chat(messages)

def parse_json(output):
    data = json.loads(output["content"])
    if "name" not in data:
        raise ValueError("Missing required field: name")
    return data

# Run with automatic error injection on failure
retry = StructuredRetry(max_attempts=3)
result = retry.run(
    messages=[{"role": "user", "content": "Return JSON with a 'name' field."}],
    call_fn=call_llm,
    parse_fn=parse_json,
)
print(result.value)     # parsed value
print(result.attempts)  # 1-3

# Convenience shortcut
result = structured_retry(messages, call_llm, parse_json, max_attempts=3)

# On-retry callback
def log(attempt):
    print(f"Retry {attempt.attempt}: {attempt.error}")

StructuredRetry(max_attempts=3, on_retry=log).run(messages, call_llm, parse_json)
```

## How it works

`parse_fn` is your validator. If it returns a value, that value is the result.
If it raises **any** exception, the run is treated as a failure: the error
string is captured, optionally reported to your `on_retry` callback, and (unless
attempts are exhausted) injected back into the conversation before the next
call. The original `messages` list you pass in is never mutated — each attempt
builds a fresh copy.

## API

### `StructuredRetry(max_attempts=3, error_prefix=..., instruction_suffix=..., on_retry=None)`

Reusable retry runner.

| Parameter | Default | Description |
| --- | --- | --- |
| `max_attempts` | `3` | Maximum number of LLM calls. Must be `>= 1`, otherwise `ValueError`. |
| `error_prefix` | `"Your previous response was invalid. Error: "` | Text placed before the validation error in the injected message. |
| `instruction_suffix` | `" Please try again and fix the issue."` | Text placed after the validation error. |
| `on_retry` | `None` | Optional `Callable[[RetryAttempt], None]` invoked on every failed attempt (including the final one). |

**`StructuredRetry.run(messages, call_fn, parse_fn) -> RetryResult`**

Run the retry loop. `call_fn(messages)` returns the raw LLM output;
`parse_fn(raw)` validates it and returns the parsed value or raises. Returns a
`RetryResult` on success, raises `StructuredRetryExhausted` if every attempt
fails. If a raw output is a dict containing a `content` key, it is echoed back
into the conversation as the assistant turn before the error message is
injected.

**`StructuredRetry.wrap(call_fn, parse_fn) -> Callable[[messages], RetryResult]`**

Bind `call_fn` and `parse_fn` and return a callable that only needs `messages`.

### `structured_retry(messages, call_fn, parse_fn, max_attempts=3, ...) -> RetryResult`

One-shot convenience function. Accepts the same keyword options as
`StructuredRetry` (`error_prefix`, `instruction_suffix`, `on_retry`).

### `RetryResult`

| Attribute | Description |
| --- | --- |
| `value` | The parsed value returned by `parse_fn`. |
| `attempts` | 1-based number of the attempt that succeeded. |
| `history` | List of `RetryAttempt` records for each failed attempt. |
| `ok` | Always `True` (a `RetryResult` is only returned on success). |

### `RetryAttempt`

A record of one failed attempt: `attempt` (int), `raw_output` (the raw LLM
output), and `error` (the validation error string).

### `StructuredRetryExhausted`

Raised when all attempts fail. Exposes `attempts` (int) and `last_error` (str).

## Development

Run the test suite with the standard library — no third-party dependencies are
required:

```bash
python3 -m unittest discover -s tests
```

## License

MIT
