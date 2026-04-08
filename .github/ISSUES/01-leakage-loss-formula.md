---
title: "Fix leakage loss formula that cancels out mass flow dependency"
priority: HIGH
labels: bug, high-priority
file: env/physics.py:100-105
---

## Problem Description
The leakage loss formula `(m_dot * Uc * U2) / (2 * m_dot + 1e-6)` algebraically cancels out the mass flow variable `m_dot`, making the loss independent of mass flow. This is physically incorrect as leakage losses should vary with mass flow rate.

When `m_dot >> 1e-6`, the formula simplifies to approximately `(Uc * U2) / 2`, completely removing mass flow dependence.

## Current Code
```python
def leakage_loss(params):
    m_dot = params["m_dot"]
    Uc = params["Uc"]
    U2 = params["U2"]
    return (m_dot * Uc * U2) / (2 * m_dot + 1e-6)
```

## Impact
- Leakage loss remains nearly constant regardless of mass flow rate
- Physics model produces incorrect efficiency calculations
- RL agent receives inaccurate reward signals

## Suggested Fix
Replace with a physically accurate formula that properly depends on mass flow. Consider using a standard orifice flow model or seal leakage formula from turbomachinery literature.
