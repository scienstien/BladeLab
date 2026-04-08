---
title: "Add a checked-in env template and document runtime configuration"
priority: HIGH
labels: documentation, developer-experience, configuration
files:
  - .env.example
  - README.md
  - .gitignore
  - inference.py
  - api/routes.py
---

## Summary

The repo reads OpenAI configuration from environment variables at runtime, but it does not provide a checked-in `.env.example` or clear documentation for which variables are required and which are optional.

## Current State

- `.env` is ignored by git via `.gitignore`
- `inference.py` reads:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL` or `API_BASE_URL`
  - `MODEL_NAME`
- `README.md` does not document this runtime env contract
- `.env.example` does not exist

## Why This Matters

- New contributors do not have a canonical env template
- Deployment setup is harder than it needs to be because the required runtime variables are implicit in code
- The expected boundary between checked-in config and local secrets is undocumented

## Proposed Fix

- Add `.env.example` with placeholder values only
- Document the runtime env contract in `README.md`
- State clearly that `.env` stays local and is not checked in
- Keep the documented variable set minimal and aligned with actual code paths

## Acceptance Criteria

- [ ] Add `.env.example` with placeholders for:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL` or `API_BASE_URL`
  - `MODEL_NAME`
- [ ] Update `README.md` with a setup section covering local env configuration
- [ ] Document which variables are required versus optional
- [ ] Ensure no real secrets are added to tracked files
