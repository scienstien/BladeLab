import math


# -------------------------------
# GEOMETRY (Bayomi-inspired)
# -------------------------------
INIT_PARAMS = {
    # rotational
    "rpm": 55000,

    # geometry
    "r1": 0.038,        # inlet hub radius (m)
    "r2": 0.08,         # outlet radius (m)
    "b2": 0.012,        # outlet width (m)
    "Z": 7,             # blade count

    # derived diameters (keep consistent)
    "D1": 0.076,        # 2 * r1
    "D2": 0.16,         # 2 * r2

    # blade angles
    "blade_angle": 60.0,
    "alpha2": 60.0,     # exit flow angle

    # thermodynamics
    "Cp": 1005,
    "T1": 300,
    "T2": 320,          # small temp rise guess

    # fluid
    "rho1": 1.2,
    "rho2": 1.1,
    "mu": 1.8e-5,

    # leakage / clearance
    "clearance": 0.0003,

    # placeholders (will be computed dynamically)
    "U2": None,
    "U1": None,
    "W1": None,
    "W2": None,
    "Cu2": None,
    "Ca": None,
    "Uc": None
}

# Bounds
BOUNDS = {
    "r2": (0.05, 0.2),
    "blade_angle": (35, 80),
    "b2": (0.005, 0.03),
    "Z": (5, 20)
}

# Targets
PR_TARGET = 2.5

# Constraint limits (will evolve later)
SURGE_LIMIT = 0.75
CHOKE_LIMIT = 1.05

# Loss coefficients (tune later)
COEFFS = {
    "k_bl": 0.6,
    "k_inc": 0.7,
    "k_sf": 0.004,
    "k_leak": 0.02,
    "k_pr": 2.0,
    "k_surge": 5.0,
    "k_choke": 5.0
}