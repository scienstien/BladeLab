---
title: "OpenAIPolicy lacks retry/backoff and structured failure handling"
priority: MEDIUM
labels: bug, enhancement, medium-priority
file: inference.py:167-190
---

## Problem Description
The `OpenAIPolicy` class in `inference.py` lacks robust error handling for API failures:
- No retry logic for transient failures (timeouts, rate limits)
- Fragile response parsing that may break on unexpected API response formats
- Immediate fallback to default action without attempting recovery

## Current Code
```python
def get_action(self, state):
    try:
        response = self.client.responses.create(...)
        action_text = getattr(response, "output_text", "")  # Fragile parsing
        return self._parse_action(action_text)
    except AuthenticationError:
        log.error("OpenAI auth failed")
        return safe_default_action()
    except RateLimitError:
        log.error("OpenAI rate limited")
        return safe_default_action()
    # ... other error handlers, but NO retry logic
```

## Impact
- Transient API failures cause immediate fallback to random/default actions
- No resilience against temporary network issues or rate limiting
- Wasted API credits on failed requests that could succeed with retry

## Acceptance Criteria
- [ ] **Bounded retries**: Implement exponential backoff with max 3-5 retries
- [ ] **Explicit malformed-output handling**: Handle cases where API returns unexpected structure
- [ ] **Tests/mocks** for the following scenarios:
  - [ ] Timeout errors (request takes >30s)
  - [ ] Rate-limit errors (429 response)
  - [ ] Empty response (API returns success but no content)
  - [ ] Invalid JSON (malformed response body)
- [ ] **Graceful degradation**: After retries exhausted, log detailed error before fallback

## Suggested Implementation
```python
def get_action(self, state):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = self.client.responses.create(...)
            action_text = self._extract_output_text(response)
            return self._parse_action(action_text)
        except (RateLimitError, TimeoutError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            raise
    log.error(f"OpenAI failed after {MAX_RETRIES} attempts: {last_error}")
    return safe_default_action()
```
