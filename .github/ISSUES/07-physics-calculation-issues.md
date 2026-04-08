---
title: "Fix multiple physics calculation issues"
priority: MEDIUM
labels: bug, medium-priority
file: env/physics.py
---

## Problem Description
Multiple physics calculations have potential issues:

### 1. Pressure ratio formula (line 149-153)
May overestimate PR by applying isentropic relation to lossy compression without accounting for irreversibilities.

### 2. Skin friction singularity (line 69)
When `beta_m` approaches 90°, `cos(beta_m)` approaches 0, causing `path_length` to approach infinity.

### 3. Disk friction formula (lines 36-46)
Dimensional analysis suggests potential transcription error in the formula.

### 4. Hardcoded blade angle (line 65)
`beta1 = math.radians(30.0)` is hardcoded instead of using state parameters.

## Current Code
```python
# Pressure ratio - may overestimate
def compute_pressure_ratio(head, losses, params):
    gamma = 1.4
    term = 1 + (head - losses) / (params["Cp"] * params["T1"])
    term = max(1e-6, term)
    return term ** ((gamma - 1) / gamma)

# Skin friction - singularity at 90°
beta1 = math.radians(30.0)  # Hardcoded
beta2 = math.radians(params["blade_angle"])
beta_m = (2 * beta2 + beta1) / 3.0
path_length = (r2 - r1) / math.cos(beta_m)  # Singularity!

# Disk friction - dimensional concerns
def disk_friction_loss(params):
    rho = params["rho1"]
    mu = params["mu"]
    U2 = params["U2"]
    r2 = params["r2"]
    term1 = 0.0402 * rho * (U2 / r2)**3 * (r2**5)
    term2 = (U2 * r2 * rho / mu)**0.2
    return term1 / (term2 * m_dot)
```

## Impact
- Pressure ratio calculations may be optimistic
- Numerical instability when blade angles approach 90°
- Disk friction losses may be incorrectly scaled
- Hardcoded values reduce model flexibility

## Suggested Fix
1. Review pressure ratio formula against thermodynamic references
2. Add bounds checking for `beta_m` or use alternative path length formulation
3. Verify disk friction formula dimensions against turbomachinery references
4. Replace hardcoded `beta1` with computed value from state
