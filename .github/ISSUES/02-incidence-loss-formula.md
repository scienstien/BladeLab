---
title: "Fix incidence loss formula missing angle-of-attack dependence"
priority: HIGH
labels: bug, high-priority
file: env/physics.py:29-33
---

## Problem Description
The incidence loss formula `F_mc * (W1**2) / 2` is purely velocity-based and lacks any dependence on the angle of attack (difference between flow angle and blade angle). Incidence losses physically occur when the incoming flow angle does not match the blade inlet angle.

## Current Code
```python
def incidence_loss(params):
    W1 = params["W1"]
    F_mc = COEFFS["k_inc"]
    return F_mc * (W1**2) / 2
```

## Impact
- Incidence loss does not capture the physical phenomenon it's meant to model
- Designs with poor blade-flow angle matching are not penalized correctly
- Optimization may converge to designs with high incidence losses

## Suggested Fix
The formula should depend on the incidence angle `i = alpha1 - beta1` (flow angle minus blade angle):
```python
def incidence_loss(params):
    alpha1 = params["alpha1"]
    beta1 = params["beta1"]
    W1 = params["W1"]
    F_mc = COEFFS["k_inc"]
    incidence_angle = alpha1 - beta1
    return F_mc * W1**2 * (1 - math.cos(incidence_angle))
```
