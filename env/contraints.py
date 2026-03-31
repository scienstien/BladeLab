from env.config import SURGE_LIMIT, CHOKE_LIMIT

def check_constraints(physics, m_max):
    m = physics["mass_flow"]
    phi = m / (m_max + 1e-6)

    surge_margin = max(0.0, 0.75 - phi)
    choke_margin = max(0.0, phi - 1.05)

    return {
        "phi": phi,
        "surge": phi < SURGE_LIMIT,
        "choke": phi > CHOKE_LIMIT,
        "surge_margin": surge_margin,
        "choke_margin": choke_margin,
    }
