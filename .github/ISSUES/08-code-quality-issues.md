---
title: "Address code quality and maintainability issues"
priority: LOW
labels: enhancement, low-priority
files:
  - env/core_env.py:86
  - env/constraints.py:26-27
  - env/physics.py:81
  - inference.py:382
---

## Problem Description
Several code quality issues affect maintainability:

### 1. Redundant constraint check (core_env.py:86)
History stores redundant constraint computation instead of reusing already-computed values.

### 2. Confusing margin naming (constraints.py:26-27)
`SURGE_LIMIT` and `CHOKE_LIMIT` naming may be confusing - positive surge_margin means above limit (good), but naming suggests otherwise.

### 3. Clearance loss variable naming (physics.py:81)
Inconsistent variable naming (`rs1`, `rh1`) - `rs1 = params["r1"]` but why `rs1`?

### 4. Memory concern (inference.py:382)
Storing all episodes in `evaluate_agent` may cause memory issues for large evaluations.

## Current Code
```python
# Redundant constraint check in history
self.history.append({
    "step": self.step_count,
    "state": self._build_obs(self.prev_physics, check_constraints(self.prev_physics)).model_dump(),
})

# Confusing margin naming
surge_margin = mass_flow - surge_limit
choke_margin = effective_choke_limit - mass_flow

# Clearance loss inconsistent naming
rs1 = params["r1"]  # Why rs1?
rh1 = 0.0           # Hardcoded to 0?

# Memory concern
summary = {
    "episodes": episode_results,  # Stores full episode data
}
```

## Impact
- Increased memory usage for large evaluations
- Harder to maintain and understand code
- Potential confusion for new developers

## Suggested Fix
1. Remove redundant `check_constraints()` call in history storage
2. Add comments clarifying margin sign conventions or rename variables
3. Use consistent variable naming in clearance loss
4. Make episode storage optional or limit stored history size
