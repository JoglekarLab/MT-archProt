# =============================================================
# parameters.py
# =============================================================
# All simulation parameters for microtubule dynamics with EB1.
# Imported by initialization.py and simulation.py.
#
# Usage:
#   from parameters import *
#
# Naming conventions:
#   kon_    = on-rate (association)
#   koff_   = off-rate (dissociation)
#   k_      = other rate constant
#   conc_   = concentration
#   _tub    = tubulin
#   _prot   = EB1 protein
#   _GTP / _GDP = nucleotide state of relevant dimer
#
#   Protein site types:
#     _0 = lattice            (4 surrounding tubulins)
#     _1 = lateral edge       (2 tubulins on adjacent PFs, exposed side)
#     _2 = longitudinal edge  (2 tubulins on same PF, protofilament tip)
#     _3 = 3-tubulin edge     (3 surrounding tubulins)

#
# Units:
#   kon_tub_ : uM-1 s-1 (per micromolar tubulin per second per PF)
#   kon_prot_: nM-1 s-1 (per nanomolar EB1 per second per site)
#   koff_    : s-1
#   conc_    : uM (micromolar)
#   time     : seconds
# =============================================================


# -------------------------------------------------------------
# SIMULATION SETTINGS
# -------------------------------------------------------------
n_pf            = 13        # number of protofilaments
                            # MATLAB: filN = 13

n_iterations    = 450000    # total Gillespie steps
                            # MATLAB: nIterations = 450000

seed_length     = 25        # starting seed length in dimers
                            # MATLAB: startL = 25, fixedSeedSize = 25

array_len_init  = 300       # initial pre-allocated height of MT arrays (dimers)
                            # MATLAB: startArrayLen = 300

snapshot_freq   = 1000      # steps between output snapshots
                            # 1-2 seconds of real simulation time per snapshot
                            # MATLAB: cortime = 1000

taper_threshold = 75        # max allowed taper (dimers) before catastrophe trigger
                            # if longest PF exceeds BOTH shortest PFs by this amount,
                            # lateral_break_fold_factor is set to 1000
                            # MATLAB: LongestTaperLength = 75


# -------------------------------------------------------------
# CONCENTRATIONS
# -------------------------------------------------------------
conc_tubGTP = 12.0          # free GTP-tubulin concentration (uM)
                            # MATLAB: tubGtpC = 12

conc_prot   = 0.200         # EB1 concentration (uM)
                            # MATLAB: protConcUM = 0.200
                            # Convert to nM inline where needed: conc_prot * 1e3


# -------------------------------------------------------------
# TUBULIN ON-RATE
# -------------------------------------------------------------
kon_tub = 0.65              # tubulin on-rate per protofilament (uM-1 s-1)
                            # MATLAB: kPlus = kPlusMT / filN = 0.65
                            # Derived from experimentally measured whole-MT rate:
                            #   kPlusMT = 0.65 * 13 = 8.45 uM-1 s-1 MT-1
                            # Effective rate used in Gillespie:
                            #   t = -log(rand) / (kon_tub * conc_tubGTP)

# -------------------------------------------------------------
# TUBULIN OFF-RATES
# -------------------------------------------------------------
# The off-rate of a tip dimer depends on the nucleotide state of the
# dimer directly below it along the protofilament, not the tip dimer itself.
# The dimer below determines the strength of the longitudinal bond being
# broken during dissociation.
# GTP below = stable longitudinal bond = slow off-rate
# GDP below = weak longitudinal bond   = fast off-rate

koff_tubGTP = 0.2           # off-rate when dimer below tip is GTP (s-1)
                            # MATLAB: kshorten_TBelow = 0.2

koff_tubGDP = 200.0         # off-rate when dimer below tip is GDP (s-1)
                            # MATLAB: kshorten_DBelow = 200
                            # ~1000x faster than koff_tubGTP


# -------------------------------------------------------------
# GTP HYDROLYSIS
# -------------------------------------------------------------
k_hydrolysis = 0.55 / n_pf  # GTP hydrolysis rate per protofilament (s-1)
                             # MATLAB: kHyd = 0.55 / filN = 0.0423

# -------------------------------------------------------------
# LATERAL BOND FORMATION RATES
# -------------------------------------------------------------
# Formation is nucleotide-independent in MATLAB program!

k_lateralbond      = 100.0  # lateral bond formation, B-lattice (s-1)
                            # MATLAB: klatbond = 100

k_lateralbondSeam  = 100.0  # lateral bond formation (s-1)
                            # MATLAB: klatbondseam = 100
                            # alpha-beta contact instead of beta-beta!


# -------------------------------------------------------------
# LATERAL BOND BREAKAGE RATES
# -------------------------------------------------------------
# _TT = both tubulins at the lateral interface are GTP
# _TD = one GTP one GDP
# _DD = both GDP
# Seam rates are higher (weaker) because contacts are less complementary

# Lattice (between PF1-PF12 and their right neighbors)
k_lateralbreak_TT       = 70.0   # s-1, MATLAB: klatbreak_TT = 70
k_lateralbreak_TD       = 90.0   # s-1, MATLAB: klatbreak_TD = 90
k_lateralbreak_DD       = 400.0  # s-1, MATLAB: klatbreak_DD = 400

# Seam (between PF13 and PF1, weaker contacts)
seam_weaken_factor = 2.0 # "half neighbor" as 2x weaker rates
k_lateralbreakSeam_TT   = k_lateralbreak_TT * seam_weaken_factor   # s-1, MATLAB: klatbreakseam_TT = 140
k_lateralbreakSeam_TD   = k_lateralbreak_TD * seam_weaken_factor   # s-1, MATLAB: klatbreakseam_TD = 180
k_lateralbreakSeam_DD   = k_lateralbreak_DD * seam_weaken_factor   # s-1, MATLAB: klatbreakseam_DD = 1200


# -------------------------------------------------------------
# LATERAL BOND STABILIZATION
# -------------------------------------------------------------
lateral_stabilization_factor = 10.0
# When a tubulin subunit has lateral bonds on BOTH sides simultaneously its breakage rate
# is divided by this factor, making it harder to break.
# Reflects cooperative stabilization from having neighbors on both sides.
# Applies to:
#   PF2-12:  when left bond AND right bond both exist
#   PF1:     when both upper and lower seam bonds exist
#   PF13:    when both bonds to PF12 and PF1 exist
# actual_rate = k_lateralbreak / lateral_stabilization_factor
# MATLAB: Pibreak = 10

# -------------------------------------------------------------
# INTER-PROTEIN BOND PARAMETERS
# -------------------------------------------------------------
interaction_range              = 1      # 0 = same height only, 1 = ±1 height
k_prot_bond_form_same_h        = 100.0/5  # s-1, 100.0 is k_lateralbond for tubulin. Same height bond formation rate
k_prot_bond_form_diff_h        = 20.0/5  # s-1, 100.0 is k_lateralbond for tubulin. ±1 height bond formation rate (slower)

k_prot_bond_break              = 70.0/5   # s-1, 70.0 is k_lateralbreak_TT (nucleotide-independent)
prot_bond_stabilization_factor = 10.0/5   # 10 is lateral_stabilization_factor for tubulin

# -------------------------------------------------------------
# EB1 PROTEIN ON-RATES  (nM-1 site-1 s-1)
# -------------------------------------------------------------
# Nucleotide state does NOT affect on-rate.
# The rate difference between site types is due to steric hindrance.
# A closed 4-tubulin pocket blocks access; an open edge site allows easy entry.
#
# Effective arrival rate used in Gillespie:
#   t = -log(rand) / (kon_prot_X * conc_prot * 1e3 * n_sites_of_type_X)
# conc_prot * 1e3 converts uM to nM

kon_prot_0  = 0.000023  # lattice (4 tubulins), ~70x slower than edge
                        # MATLAB: kProtLattice = 0.000023

# ******** 2-tubulin edge and 3-tubulin edges are lumped together, should check ********

kon_prot_1  = 0.0016    # lateral edge (2 horizontal, adjacent PFs)
                        # MATLAB: kProtEdge = 0.0016
                        
kon_prot_2  = 0.0016    # longitudinal edge (2 vertical, same PF)
                        # MATLAB: kProtEdge = 0.0016

kon_prot_3  = 0.0016    # 3-tubulin edge
                        # MATLAB: kProtEdge = 0.0016
                        # expected between kon_prot_0 and kon_prot_2-edges


# -------------------------------------------------------------
# EB1 PROTEIN OFF-RATES  (s-1)
# -------------------------------------------------------------
# Both site type AND nucleotide state of the pocket affect koff.
# More tubulin contacts = more bonds = slower koff
# GTP state = better shape complementarity = slower koff
#
# Pocket nucleotide state encoding (stored in prot_sites layer 1):
#   0 = both tubulins GTP  -> use _GTP rates
#   1 = mixed GTP/GDP      -> use _GDP rates (conservative)
#   2 = both tubulins GDP  -> use _GDP rates
#
# Experimental support:
#   _0 GTP/GDP : well measured from lattice dwell times
#   _1 GTP/GDP : inferred from ~20ms short dwell component (Reid 2019)
#   _2 GTP/GDP : currently assumed same as _1
#   _3 GTP/GDP : currently assumed same as _1, but may be lower due to more contacts

koff_prot_GTP_0 = 0.29      # lattice, GTP pocket
                            # MATLAB: kPOffLatticeGTP = 0.29
                            # long dwell ~150ms (Reid 2019)

koff_prot_GDP_0 = 1.7       # lattice, GDP pocket
                            # MATLAB: kPOffLatticeGDP = 1.7

koff_prot_GTP_1 = 2.9       # lateral edge, GTP pocket
                            # MATLAB: kPOffEdgeGTP = 2.9
                            # short dwell ~20ms (Reid 2019)

koff_prot_GDP_1 = 25.0      # lateral edge, GDP pocket
                            # MATLAB: kPOffEdgeGDP = 25
                            
koff_prot_GTP_2 = 2.9       # longitudinal edge, GTP pocket
                            # MATLAB: same as _1

koff_prot_GDP_2 = 25.0      # longitudinal edge, GDP pocket
                            # MATLAB: same as _1

koff_prot_GTP_3 = 2.9       # 3-tubulin edge, GTP pocket
                            # MATLAB: same as _1

koff_prot_GDP_3 = 25.0      # 3-tubulin edge, GDP pocket
                            # MATLAB: same as _1

# =============================================================
# PROTEIN BINDING SITE TYPES
# =============================================================
SITE_LATTICE = 0    # 4 surrounding tubulins
                    # MATLAB internal value: 0

SITE_EDGELAT2 = 1   # 2-edge lateral/horizontal
                    # MATLAB internal value: 3

SITE_EDGELONG2 = 2  # 2-edge longitudinal/vertical
                    # best-supported edge binding geometry?????????
                    # MATLAB internal value: 2

SITE_EDGE3    = 3   # 3 surrounding tubulins
                    # MATLAB internal value: 3
                    
SITE_SINGLE   = 4   # 1 surrounding tubulin
                    
SITE_SEAM     = -1  # Seam groove. No binding in this simulation.

SITE_EMPTY    = -9  # For cases where site_single looses a tubulin, so needs to be updated. Basically, nothing can bind here, just in SITE_SINGLE, as there is no tubulin.

# =============================================================
# NUCLEOTIDE STATE CONSTANTS
# =============================================================
NUC_GTP   = 0   # both tubulins at pocket are GTP  -> use _GTP koff
NUC_MIXED = 1   # one GTP, one GDP                 -> use _GDP koff (conservative)
NUC_GDP   = 2   # both tubulins are GDP            -> use _GDP koff

# =============================================================
# QUICK REFERENCE TABLE
# =============================================================
#
# Python name                   | MATLAB name               | Value
# ------------------------------|---------------------------|------------------
# n_pf                          | filN                      | 13
# n_iterations                  | nIterations               | 450000
# seed_length                   | startL / fixedSeedSize    | 25
# array_len_init                | startArrayLen             | 300
# snapshot_freq                 | cortime                   | 1000
# taper_threshold               | LongestTaperLength        | 75
# conc_tubGTP                   | tubGtpC                   | 12 uM
# conc_prot                     | protConcUM                | 0.200 uM
# kon_tub                       | kPlus                     | 0.65 uM-1 s-1 PF-1
# koff_tubGTP                   | kshorten_TBelow           | 0.2 s-1
# koff_tubGDP                   | kshorten_DBelow           | 200 s-1
# k_hydrolysis                  | kHyd                      | 0.0423 s-1
# k_lateralbond                 | klatbond                  | 100 s-1
# k_lateralbondSeam             | klatbondseam              | 100 s-1
# k_lateralbreak_TT             | klatbreak_TT              | 70 s-1
# k_lateralbreak_TD             | klatbreak_TD              | 90 s-1
# k_lateralbreak_DD             | klatbreak_DD              | 400 s-1
# k_lateralbreakSeam_TT         | klatbreakseam_TT          | 140 s-1
# k_lateralbreakSeam_TD         | klatbreakseam_TD          | 180 s-1
# k_lateralbreakSeam_DD         | klatbreakseam_DD          | 1200 s-1
# lateral_stabilization_factor  | Pibreak                   | 10
# lateral_break_fold_factor     | LateralBreakingFoldFactor | 1.0 (usually, but set to 1000 if taper threshold exceeded)
# kon_prot_0                    | kProtLattice              | 0.000023 nM-1 s-1
# kon_prot_1                    | kProtEdge                 | 0.0016 nM-1 s-1 
# kon_prot_2                    | kProtEdge                 | 0.0016 nM-1 s-1 *
# kon_prot_3                    | kProtEdge                 | 0.0016 nM-1 s-1 *
# koff_prot_GTP_0               | kPOffLatticeGTP           | 0.29 s-1
# koff_prot_GDP_0               | kPOffLatticeGDP           | 1.7 s-1
# koff_prot_GTP_1               | kPOffEdgeGTP              | 2.9 s-1
# koff_prot_GDP_1               | kPOffEdgeGDP              | 25.0 s-1
# koff_prot_GTP_2               | kPOffEdgeGTP              | 2.9 s-1 *
# koff_prot_GDP_2               | kPOffEdgeGDP              | 25.0 s-1 *
# koff_prot_GTP_3               | kPOffEdgeGDP              | 2.9 s-1 *
# koff_prot_GDP_3               | kPOffEdgeGDP              | 25.0 s-1 *
#
# * = need to check
