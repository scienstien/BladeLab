---
title: "Minor issues and improvements backlog"
priority: LOW
labels: enhancement, low-priority
files:
  - env/tasks.py:68-81
  - env/constraints.py:26-27
  - env/physics.py:65, 81
---

## Problem Description
Several minor issues that should be addressed:

### 1. Reward structure can be negative for feasible designs (env/tasks.py:68-81)
Feasible designs can receive negative rewards, which may confuse the agent.

### 2. Missing edge case tests
No tests for division by zero, NaN handling, or extreme values in physics calculations.

### 3. Confusing margin naming (env/constraints.py:26-27)
With `SURGE_LIMIT=0.75` and `CHOKE_LIMIT=1.05`, the naming is counterintuitive.

### 4. Reward structure issues (env/tasks.py:68-81)
Even when feasible (+3.0), the quadratic penalty can exceed this if `pr_error` is large.

### 5. Hardcoded blade angle (env/physics.py:65)
`beta1 = math.radians(30.0)` hardcoded instead of using parameters.

### 6. Clearance loss variable naming (env/physics.py:81)
`rs1 = params["r1"]` - inconsistent naming with `r2`.

## Impact
- Agent may receive confusing reward signals
- Edge cases may cause runtime errors
- Harder to maintain and understand code

## Suggested Fix
1. Review reward structure to ensure feasible designs always receive positive base reward
2. Add edge case tests for physics calculations
3. Improve variable naming consistency
4. Document margin sign conventions
