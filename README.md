# llm-structured-retry-py

Retry LLM calls by injecting validation errors back into the conversation. More effective than silent retries because the model sees what went wrong.

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

## License

MIT
