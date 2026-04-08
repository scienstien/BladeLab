---
title: "Fix recirculation loss formula that duplicates blade loading loss"
priority: HIGH
labels: bug, high-priority
file: env/physics.py:108-130
---

## Problem Description
The recirculation loss formula (lines 108-130) appears to be a copy-paste from blade_loading_loss (lines 7-26). Lines 123-128 in recirculation_loss are nearly identical to lines 20-25 in blade_loading_loss, using the same `term1`, `term2`, `term3`, `term4` structure.

Recirculation loss should have a distinct physical formula modeling flow separation and secondary flows.

## Current Code
```python
def recirculation_loss(params):
    W1 = params["W1"]
    W2 = params["W2"]
    U2 = params["U2"]
    D1 = params["D1"]
    D2 = params["D2"]
    z = params["Z"]
    Cp = params["Cp"]
    T1 = params["T1"]
    T2 = params["T2"]
    alpha2 = math.radians(params["alpha2"])

    term1 = 1 - W2 / W1
    term2 = (Cp * (T2 - T1)) / (U2**2)
    term3 = W1 / U2
    term4 = z / math.pi * (1 - D1 / D2) + 2 * D1 / D2
    core = term1 + term2 / (term3 * term4)
    return 0.02 * math.tan(alpha2) * (core**2) * U2**2
```

## Impact
- Recirculation loss is not modeled correctly
- Double-counting of blade loading effects
- Physics model lacks distinct penalty for recirculation/separation phenomena

## Suggested Fix
Implement a proper recirculation loss formula based on turbomachinery literature. Recirculation losses typically depend on flow coefficient deviation from design point, blade loading parameters, and diffusion factor.
