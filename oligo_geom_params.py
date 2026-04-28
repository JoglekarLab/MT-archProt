# =============================================================
# oligo_geom_params.py
# =============================================================
# Oligomer geometry and diffusion scaling parameters.
# Computes diffusion scaling factors (alpha_k) for each oligomer
# size based on the prolate spheroid model.
#
# Model:
#   Dk = D1 * k^(-1/3) * Ft(p1) / Ft(pk)
#   where pk = p1 / k  (oligomer grows along long axis)
#   and Ft(p) is the Perrin translational friction factor.
#
# Reference: Zhou (1997), Gillespie (2009), Sept & McCammon (2001)
# =============================================================

import numpy as np

# -------------------------------------------------------------
# GEOMETRY
# -------------------------------------------------------------
p1_oligomer = 0.285     # axial ratio (b/a) of monomer — moderate prolate spheroid
                        # p=1 → sphere, p→0 → needle
max_oligomer_size = 6   # maximum oligomer size (hexamer)

# -------------------------------------------------------------
# OLIGOMER DISTRIBUTION IN SOLUTION (static, fractions by subunit count)
# -------------------------------------------------------------
# Must sum to 1. Weighted by subunit, not by oligomer count.
# Mostly dimers and trimers as discussed.
oligomer_distribution = {
    1: 0.10,   # monomers
    2: 0.35,   # dimers
    3: 0.30,   # trimers
    4: 0.15,   # tetramers
    5: 0.07,   # pentamers
    6: 0.03,   # hexamers
}

assert abs(sum(oligomer_distribution.values()) - 1.0) < 1e-6, \
    "oligomer_distribution must sum to 1.0"

# -------------------------------------------------------------
# PERRIN TRANSLATIONAL FRICTION FACTOR
# -------------------------------------------------------------
def _perrin_Ft(p: float) -> float:
    """
    Perrin translational friction factor for a prolate spheroid
    with axial ratio p = b/a < 1.
    Ft(p) = sqrt(1-p^2) / (p^(2/3) * ln((1 + sqrt(1-p^2)) / p))
    """
    sq = np.sqrt(1.0 - p**2)
    return sq / (p**(2/3) * np.log((1.0 + sq) / p))

# -------------------------------------------------------------
# DIFFUSION SCALING FACTORS AND ARRIVAL PROBABILITIES
# -------------------------------------------------------------

f_long = 0.7   # fraction of growth along long axis (0=isotropic, 1=fully elongated)
# Assuming the oligomer grows 70% in the long axis direction (linear chain growth).
def _alpha_k(k: int) -> float:
    pk = p1_oligomer * k**(-f_long + 0.5*(1 - f_long)*2/1)
    # simplified: pk = p1 * k^(-(2f-1)/2) = p1 * k^(-0.55) for f=0.7
    pk = p1_oligomer * k**(-0.55)
    return k**(-1/3) * _perrin_Ft(p1_oligomer) / _perrin_Ft(min(pk, 0.999))

# oligomer_alpha[k] = Dk/D1, the diffusion scaling factor for size k
# kon for oligomer k = kon_monomer * oligomer_alpha[k]
oligomer_alpha = {k: _alpha_k(k) for k in range(1, max_oligomer_size + 1)}

# Arrival probs
arrival_weights = {k: oligomer_distribution[k] * oligomer_alpha[k] 
                   for k in range(1, max_oligomer_size + 1)}
total = sum(arrival_weights.values())
arrival_probs = {k: v / total for k, v in arrival_weights.items()}

# =============================================================
# QUICK REFERENCE
# =============================================================
# oligomer_alpha values for p1=0.285, linear chain growth:
#   k=1: 1.0000
#   k=2: 0.6623
#   k=3: 0.5070
#   k=4: 0.4155
#   k=5: 0.3544
#   k=6: 0.3103