from env.config import CHOKE_LIMIT, SURGE_LIMIT


def check_constraints(physics, surge_limit=SURGE_LIMIT, choke_limit=CHOKE_LIMIT):
    mass_flow = physics["mass_flow"]

    surge_margin = mass_flow - surge_limit
    choke_margin = choke_limit - mass_flow

    surge = mass_flow < surge_limit
    choke = mass_flow > choke_limit

    return {
        "feasible": not (surge or choke),
        "surge": surge,
        "choke": choke,
        "surge_margin": surge_margin,
        "choke_margin": choke_margin,
        "mass_flow": mass_flow,
    }
