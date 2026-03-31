"""Physics simulation module for TurboDesigner 2.0"""

from env.config import INIT_PARAMS, BOUNDS


def compute_physics(params):
    """
    Compute physics-based outputs for the current design parameters.

    Args:
        params: Dict with 'radius', 'blade_angle', 'thickness'

    Returns:
        Dict with 'efficiency', 'pressure_ratio', 'mass_flow'
    """
    radius = params.get("radius", INIT_PARAMS["radius"])
    blade_angle = params.get("blade_angle", INIT_PARAMS["blade_angle"])
    thickness = params.get("thickness", INIT_PARAMS["thickness"])

    # Normalize inputs to 0-1 range
    r_norm = (radius - BOUNDS["radius"][0]) / (BOUNDS["radius"][1] - BOUNDS["radius"][0])
    a_norm = (blade_angle - BOUNDS["blade_angle"][0]) / (BOUNDS["blade_angle"][1] - BOUNDS["blade_angle"][0])
    t_norm = (thickness - BOUNDS["thickness"][0]) / (BOUNDS["thickness"][1] - BOUNDS["thickness"][0])

    # Physics-based calculations
    # Mass flow: proportional to radius^2 and blade angle
    mass_flow = 0.5 + 0.3 * r_norm + 0.2 * a_norm

    # Pressure ratio: function of blade angle and radius
    pressure_ratio = 1.5 + 0.8 * a_norm + 0.3 * r_norm - 0.2 * t_norm

    # Efficiency: optimal at mid-range values (bell curve behavior)
    eff_r = 1.0 - 0.3 * ((r_norm - 0.5) ** 2)
    eff_a = 1.0 - 0.3 * ((a_norm - 0.5) ** 2)
    eff_t = 1.0 - 0.2 * ((t_norm - 0.5) ** 2)
    efficiency = 0.7 * eff_r * eff_a * eff_t

    return {
        "efficiency": efficiency,
        "pressure_ratio": pressure_ratio,
        "mass_flow": mass_flow
    }


class Physics:
    """Handles physics calculations and simulations."""

    def __init__(self):
        pass

    def calculate_force(self, mass, acceleration):
        """Calculate force using F = ma."""
        return mass * acceleration

    def calculate_velocity(self, initial_velocity, acceleration, time):
        """Calculate final velocity."""
        return initial_velocity + acceleration * time

    def calculate_position(self, initial_position, velocity, time):
        """Calculate new position."""
        return initial_position + velocity * time
