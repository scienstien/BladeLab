---
title: "Document deployment-time secret injection without baking secrets into the image"
priority: MEDIUM
labels: documentation, deployment, docker
files:
  - README.md
  - Dockerfile
---

## Summary

The repository has a generic Dockerfile and a basic README, but it does not explain how runtime secrets should be supplied in containerized or hosted deployments. The gap is documentation, not a need to hardcode API keys into the image.

## Current State

- `Dockerfile` sets Flask runtime variables only
- OpenAI credentials are expected to come from the environment at runtime
- `README.md` does not explain how to pass those variables during deployment
- There is no checked-in deployment note for container or hosted environments

## Why This Matters

- Operators may not know how to inject `OPENAI_API_KEY` safely
- The lack of deployment guidance encourages ad hoc configuration
- It is too easy to conflate runtime configuration with image build configuration

## Proposed Fix

- Add a deployment section to `README.md` or a small `docs/deployment.md`
- Document secret injection for local runs and containerized deployments
- Keep `Dockerfile` generic unless there is a concrete platform-specific requirement
- Avoid adding secret-bearing `ENV` defaults to the image

## Acceptance Criteria

- [ ] Document how to provide `OPENAI_API_KEY` and optional model/base URL settings at runtime
- [ ] Show at least one container run example that passes env vars externally
- [ ] State explicitly that secrets should not be baked into `Dockerfile`
- [ ] Keep deployment guidance factual and platform-agnostic unless a specific target platform is adopted
