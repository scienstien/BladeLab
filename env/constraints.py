from env.config import SURGE_LIMIT, CHOKE_LIMIT

def check_constraints(physics):
    m = physics["mass_flow"]

    surge_margin = max(0.0, SURGE_LIMIT - m)
    choke_margin = max(0.0, m - CHOKE_LIMIT)

    return {
        "surge": surge_margin > 0,
        "choke": choke_margin > 0,
        "surge_margin": surge_margin,
        "choke_margin": choke_margin,
    }