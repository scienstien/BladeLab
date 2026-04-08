---
title: "Remove redundant manual validation in API routes"
priority: MEDIUM
labels: enhancement, medium-priority
file: api/routes.py:83-96, 177-198, 275-294
---

## Problem Description
Manual validation in API routes duplicates Pydantic validation that already occurs via `PredictRequest.model_validate()`, `RolloutRequest.model_validate()`, and `EvaluateRequest.model_validate()`. This creates redundant code and potential for inconsistent error messages.

## Current Code
```python
req = PredictRequest.model_validate(payload)  # Pydantic validation

details = []  # Redundant manual validation
if req.task_name not in VALID_TASKS:
    details.append(f"task_name: invalid task '{req.task_name}'...")
if req.policy_type not in VALID_POLICY_TYPES:
    details.append(f"policy_type: invalid policy...")
if req.policy_type == "openai" and not req.model_name:
    details.append("model_name: required when policy_type is 'openai'")

if details:
    return _validation_response(details)
```

## Impact
- Code duplication increases maintenance burden
- Potential for validation logic to drift between Pydantic schemas and manual checks
- Longer, harder-to-read endpoint handlers

## Suggested Fix
1. Move validation logic into Pydantic model validators using `@field_validator`
2. Remove manual validation blocks from route handlers
3. Rely on Pydantic's `ValidationError` handling
