# =============================================================
# initialization.py
# =============================================================
# Builds all data structures for the microtubule + protein simulation.
# Run once before the main Gillespie loop in simulation.py.

# The protein used here binds at the junction between two protofilaments and two longitudinal interfaces (EB1)

# Usage:
#   from initialization import *

# Outputs:
#   MT_lattice_matrix:    Shape (13, 300, 2)  13 protofilaments, 300 height positions, 2 properties per dimer.
#   prot_matrix:          Shape (14, 300, 3)  14 grooves between protofilaments, 300 height positions, 3 properties per pocket.
# =============================================================

import numpy as np
from params import *
from helpers import classify_pocket

# =============================================================
# MT_lattice_matrix:    Shape (13, 300, 2)  
# =============================================================
# Shape: (n_pf, array_len_init, 2)
#   axis 0: protofilament index, 0-based (PF0 = MATLAB PF1)
#   axis 1: height position, 0-based (0 = minus-end base)
#   axis 2:
#     layer 0: hydrolysis state       0 = GTP, 1 = GDP
#     layer 1: right lateral bond count  0, 1, or 2
#              PF0-11: bond to next PF clockwise (B-lattice); max = 1
#              PF12:   bonds across seam to PF0 (A-lattice); max = 2
# Replaces tubAProps from MATLAB. Possible tagging flag in future.
# =============================================================
MT_lattice = np.zeros((n_pf, array_len_init, 2), dtype=np.int8)


# =============================================================
# prot_matrix:         Shape (14, 300, 3)
# =============================================================
# Shape: (n_pf + 1, array_len_init, 3)
#   axis 0: groove index
#     groove g sits between PF(g-1) and PF(g).
#     groove 0  = between PF12 and PF0 (A-lattice)
#     groove 1  = between PF0 and PF1 (B-lattice)
#     ...
#     groove 12 = between PF11 and PF12 (B-lattice)
#     groove 13 = between PF12 and PF0 (A-lattice other side)
#     Note: grooves 0 and 13 are both seam contacts but at different helical offsets due to the 1.5-dimer shift at the seam. None of these two will be occupied by protein.
#   axis 1: height position
#   axis 2:
#     layer 0: site type    SITE_LATTICE / SITE_LONG2 / SITE_LAT2 / SITE_LAT3 / SITE_SINGLE
#     layer 1: nuc state    NUC_GTP / NUC_MIXED / NUC_GDP
#     layer 2: EB1 bound    0 = unoccupied, 1 = occupied
#
# MATLAB: EB1BindingPositions
# =============================================================
prot_sites = np.zeros((n_pf + 1, array_len_init, 3), dtype=np.int8)
prot_sites[0, :, 0] = SITE_SEAM
prot_sites[n_pf, :, 0] = SITE_SEAM


# =============================================================
# GLOBAL PROTOFILAMENT STATES
# =============================================================
pf_len = np.full(n_pf, seed_length, dtype=np.int32) # np.full(shape, fill_value). Length in tubulin dimers. Tubulin at height h on PF p is present only if h < pf_len[p].
                                                    # MATLAB: pfLen

highest_lat = np.full(n_pf, seed_length-1, dtype=np.int32)    # For each PF,  height of the highest subunit that is right-laterally bonded.
                                                            # Tubulin can only dissociate from at least above this height.
                                                            # MATLAB: highestFullLatMade

highest_full_GDP = seed_length - 1 # Highest height at which ALL protofilaments are GDP-hydrolyzed. Limits the inner hydrolysis check loop.
                                                           # MATLAB: highestFullHydro


# =============================================================
# PROTEIN TRACKING: prot_events
# =============================================================
# Full history of all protein binding events. MATLAB: proteinTracking
# One dict entry created at binding; updated dynamically; finalized at unbinding.
# Dict schema per entry:
# {
#   'g_idx'            : int,   groove index (0..n_pf)
#   'h'             : int,   height position
#   'site_type_on'  : int,   Site type at the time of binding
#   'nuc_state_on'  : int,   Nucleotide state at time of binding
#   't_on'          : float, simulation time of binding (seconds)
#   't_off'         : float, simulation time of unbinding (None if still bound)
#   'removal'       : str,   reason for unbinding (None if still bound)
#                            'kinetic'  = normal koff Gillespie event
#                            'tub_loss' = tubulin depolymerized under protein. 2 → 1 (LAT2/LONG2 → SINGLE) causes 'tub_loss'
#   'site_type_now' : int,   current site type (updated as MT state changes)
#   'nuc_state_now' : int,   current nuc state (updated as MT state changes)
# }
prot_events = []

# =============================================================
# PROTEIN TRACKING: bound_prots
# =============================================================
# Fast lookup of currently bound proteins by position. MATLAB: presentProteins
# Example:  bound_prots = {
#               (3, 25): 1,   # protein at groove 3, height 25 is entry 1 in prot_events
#               (8, 26): 2,   # protein at groove 8, height 26 is entry 2 in prot_events
#               }
bound_prots = {}
protein_bonds = {}              # Proteins can oligomerize forming bonds to adjacent proteins depending on the kernel.
                                # This dict tracks which proteins are bonded to which, such as:
                                # protein_bonds = {(3, 25): {(4, 25)}, (4, 25): {(3, 25), (5, 25)}, (5, 25): {(4, 25)}}

# Bonded proteins are excluded from removal pools.
n_bonded_prots_by_nuc = {
    SITE_LATTICE:   {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
    SITE_EDGELAT2:  {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
    SITE_EDGELONG2: {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
    SITE_EDGE3:     {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
}
# =============================================================
# SIMULATION
# =============================================================
time_elapsed           = 0.0   # simulation time in seconds

lateral_breaking_fold_factor = 1.0 # Catastrophe multiplier; 1.0 normally, set to 1000.0 at catastrophe.
# Set to 1000.0 when taper exceeds taper_threshold on both shortest PFs.
# Makes all lateral bonds break near-instantly, triggering rapid
# depolymerization. Reset to 1.0 when taper falls back below threshold.
# actual_rate = k_lateralbreak_XY * lateral_break_fold_factor
# MATLAB: LateralBreakingFoldFactor = 1 (normally), 1000 (catastrophe)


# =============================================================
# ADDITIONAL OUTPUTS
# =============================================================
# Lists appended every snapshot_freq steps in the simulation loop.
out_time        = []    # elapsed simulation time at each snapshot
out_pf_lengths  = []    # full pf_len array (n_pf values) at each snapshot
out_n_bound     = []    # total EB1 proteins bound at snapshot

# Bound protein counts split by site type and pocket nucleotide state.
# GTP = NUC_GTP pocket; GDP = NUC_MIXED or NUC_GDP pocket.
out_n_GTP_0     = []    # SITE_LATTICE, GTP pocket
out_n_GDP_0     = []    # SITE_LATTICE, GDP or mixed pocket
out_n_GTP_1     = []    # SITE_LAT2,   GTP pocket
out_n_GDP_1     = []    # SITE_LAT2,   GDP or mixed pocket
out_n_GTP_2     = []    # SITE_LONG2,  GTP pocket
out_n_GDP_2     = []    # SITE_LONG2,  GDP or mixed pocket
out_n_GTP_3     = []    # SITE_LAT3,   GTP pocket
out_n_GDP_3     = []    # SITE_LAT3,   GDP or mixed pocket




# =============================================================
# SEED INITIALIZATION
# =============================================================
# TUBULIN LATTICE
# All seed tubulins start as GTP (MT_lattice[:, :, 0] = 0, set by np.zeros). Every dimer in the seed is considered fully laterally bonded.
MT_lattice[:n_pf - 1, :seed_length, 1] = 1  # PF0-PF11
MT_lattice[n_pf - 1, :seed_length, 1] = 2   # PF12 (seam) has two lateral bonds per dimer in the seed, to two different tubulins on PF0 across the seam.

# PROTEIN BINDING POCKET CLASSIFICATION
# Classifies the site type of every pocket over the seed region which depends on pf_len of the current and surrounding PFs.

n_bindable_sites = {
    SITE_LATTICE:   0,
    SITE_EDGELAT2:  0,
    SITE_EDGELONG2: 0,
    SITE_EDGE3:     0,
    SITE_SINGLE:    0,
    SITE_SEAM:      0,
}

for g in range(n_pf + 1):
    for h in range(seed_length):
        site = int(prot_sites[g, h, 0])
        if site in n_bindable_sites:
            n_bindable_sites[site] += 1
            
n_bound_prots_by_nuc = {
    SITE_LATTICE:   {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
    SITE_EDGELAT2:  {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
    SITE_EDGELONG2: {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
    SITE_EDGE3:     {NUC_GTP: 0, NUC_MIXED: 0, NUC_GDP: 0},
}
            
# A groove (g) is between right PF = g,  left PF = (g-1)
#   Four surrounding tubulins:
#     right_lo: right PF at height h     present if h   < pf_len[right PF]
#     right_hi: right PF at height h+1   present if h+1 < pf_len[right PF]
#     left_lo:  left  PF at height h     present if h   < pf_len[left  PF]
#     left_hi:  left  PF at height h+1   present if h+1 < pf_len[left  PF]

# Seam grooves (g=0 and g=n_pf) are part of the seam and contain no protein.
# Classify all pockets from height 0 to seed_length-1.
for g in range(n_pf + 1):
    for h in range(seed_length):
        prot_sites[g, h, 0] = classify_pocket(g, h, pf_len)
            
# =============================================================
# VALIDATION
# =============================================================
def validate_initialization():
    """
    Sanity checks on the initialized state.
    Raises AssertionError with a descriptive message if anything fails.
    """
    assert np.all(pf_len == seed_length), \
        f"pf_len should all be {seed_length}, got: {pf_len}"

    assert np.all(highest_lat == seed_length-1), \
        f"highest_lat should all be {seed_length}, got: {highest_lat}"

    assert highest_full_GDP == seed_length - 1, \
        f"highest_hydro should be {seed_length - 1}, got: {highest_full_GDP}"

    assert np.all(MT_lattice[:, :, 0] == 0), \
        "MT_lattice hydrolysis layer should be all-GTP (0) at start"

    assert np.all(MT_lattice[:n_pf - 1, :seed_length, 1] == 1), \
        "B-lattice PF0-11 should have right lateral bond count = 1 in seed"

    assert np.all(MT_lattice[n_pf - 1, :seed_length, 1] == 2), \
        "Seam PF12 should have right lateral bond count = 2 in seed"

    assert np.all(MT_lattice[:, seed_length:, 1] == 0), \
        "No lateral bonds above seed length"

    assert np.all(prot_sites[:, :, 2] == 0), \
        "No EB1 should be bound at start"

    assert np.all(prot_sites[:, :, 1] == 0), \
        "All pocket nucleotide states should be NUC_GTP (0) at start"

    # At the blunt-seed tip (h = seed_length - 1), middle grooves (g=1..12)
    # must be SITE_LAT2: the upper two tubulins are absent, the lower two
    # are present side by side.
    tip_h = seed_length - 1
    for g in range(1, n_pf):
        site = int(prot_sites[g, tip_h, 0])
        assert site == SITE_EDGELAT2, (
            f"Middle groove g={g} at tip height h={tip_h} "
            f"should be SITE_LAT2 ({SITE_EDGELAT2}), got {site}"
        )

    # One row below the tip (h = seed_length - 2), all four tubulins are
    # present for every middle groove -> SITE_LATTICE.
    if seed_length >= 2:
        sub_tip_h = seed_length - 2
        for g in range(1, n_pf):
            site = int(prot_sites[g, sub_tip_h, 0])
            assert site == SITE_LATTICE, (
                f"Middle groove g={g} at h={sub_tip_h} "
                f"should be SITE_LATTICE ({SITE_LATTICE}), got {site}"
            )

    assert len(prot_events) == 0, "prot_events should be empty at start"
    assert len(bound_prots) == 0, "bound_prots should be empty at start"
    assert time_elapsed == 0.0,   "time_elapsed should be 0.0 at start"
    
    

    print("Initialization validation passed.")


validate_initialization()

print(f"\nInitialization complete.")
print(f"  MT_lattice shape  : {MT_lattice.shape}")
print(f"  prot_sites shape  : {prot_sites.shape}")
print(f"  pf_len            : {pf_len}")
print(f"  highest_lat       : {highest_lat}")
print(f"  highest_hydro     : {highest_full_GDP}")
print(f"  time_elapsed      : {time_elapsed}")

tip_labels = {
    SITE_SEAM:      'SEAM',
    SITE_LATTICE:   'LATTICE',
    SITE_EDGELAT2:  'EDGELAT2',
    SITE_EDGELONG2: 'EDGELONG2',
    SITE_EDGE3:     'EDGE3',
    SITE_SINGLE:    'SINGLE',
}

tip_types = [tip_labels[int(prot_sites[g, seed_length - 1, 0])]
             for g in range(n_pf + 1)]
print(f"  Tip row site types: {tip_types}")