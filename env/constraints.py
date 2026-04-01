import math

from env.config import CHOKE_LIMIT, SURGE_LIMIT

GAMMA_AIR = 1.4
DEFAULT_SLIP_FACTOR = 1.0


def compute_choke_limit(physics, gamma=GAMMA_AIR, slip_factor=DEFAULT_SLIP_FACTOR):
    rho0 = physics["rho1"]
    T0 = physics["T1"]
    gas_constant = physics["R"]
    r2 = physics["r2"]
    b2 = physics["b2"]

    a0 = math.sqrt(gamma * gas_constant * T0)
    mass_flux = rho0 * a0 * ((2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
    throat_area = slip_factor * 2.0 * math.pi * r2 * b2
    return mass_flux * throat_area


def check_constraints(physics, surge_limit=SURGE_LIMIT, choke_limit=CHOKE_LIMIT):
    mass_flow = physics["mass_flow"]
    effective_choke_limit = compute_choke_limit(physics)

    surge_margin = mass_flow - surge_limit
    choke_margin = effective_choke_limit - mass_flow

    surge = mass_flow < surge_limit
    choke = mass_flow > effective_choke_limit

    return {
        "feasible": not (surge or choke),
        "surge": surge,
        "choke": choke,
        "surge_margin": surge_margin,
        "choke_margin": choke_margin,
        "mass_flow": mass_flow,
        "choke_limit": effective_choke_limit,
    }
