"""Environment configuration and default operating point."""


INIT_PARAMS = {
    "rpm": 55000,
    "r1": 0.038,
    "r2": 0.08,
    "b2": 0.012,
    "Z": 7,
    "blade_angle": 60.0,
    "alpha2": 60.0,
    "Cp": 1005,
    "T1": 300.0,
    "T2": 320.0,
    "rho1": 1.2,
    "rho2": 1.1,
    "mu": 1.8e-5,
    "clearance": 0.0003,
}

BOUNDS = {
    "r2": (0.05, 0.2),
    "blade_angle": (35, 80),
    "b2": (0.005, 0.03),
    "Z": (5, 20),
}

PR_TARGET = 2.5

SURGE_LIMIT = 0.75
CHOKE_LIMIT = 1.05

COEFFS = {
    "k_bl": 0.6,
    "k_inc": 0.7,
    "k_sf": 0.004,
    "k_leak": 0.02,
    "k_pr": 2.0,
    "k_surge": 5.0,
    "k_choke": 5.0,
}

MAX_STEPS = 30
