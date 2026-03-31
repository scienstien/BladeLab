from env.config import COEFFS, PR_TARGET

def compute_reward(curr, prev, constraints):
    reward = 0.0

    eff = curr["efficiency"]
    prev_eff = prev["efficiency"]
    PR = curr["pressure_ratio"]

    # --- 1. Progress (PRIMARY DRIVER) ---
    reward += 10.0 * (eff - prev_eff)

    # --- 2. Useful work (PR gating) ---
    if PR < PR_TARGET:
        # heavy penalty if not doing enough compression
        reward -= 5.0 * (PR_TARGET - PR)**2
    else:
        # small bonus if above target (not too aggressive)
        reward += 0.5 * (PR - PR_TARGET)

    # --- 3. Efficiency reward (secondary, stabilizes) ---
    reward += 2.0 * eff

    # --- 4. Constraints (HARD PHYSICS) ---
    if constraints["surge"]:
        reward -= 10.0 * constraints["surge_margin"]**2

    if constraints["choke"]:
        reward -= 8.0 * constraints["choke_margin"]**2

    # --- 5. Anti-stagnation ---
    reward -= 0.02

    return reward