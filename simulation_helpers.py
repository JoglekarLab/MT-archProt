import math
import numpy as np
from params import *
from helpers import *
from initialization import *

# =============================================================
# GEOMETRY AND STATE QUERIES
# =============================================================

def tubulin_present(pf: int, h: int) -> bool:
    """
    True if tubulin dimer at protofilament pf and height h exists.
    Presence is determined entirely by pf_len.
    """
    if pf < 0 or pf >= n_pf:
        raise IndexError(f"Invalid PF index: {pf}")
    if h < 0:
        return False
    return h < pf_len[pf]

def tubulin_is_GTP(pf: int, h: int) -> bool:
    """
    True if the tubulin exists and is not hydrolyzed.
    """
    if not tubulin_present(pf, h):
        return False
    return int(MT_lattice[pf, h, 0]) == 0

def tubulin_is_GDP(pf: int, h: int) -> bool:
    """
    True if the tubulin exists and is hydrolyzed (GDP).
    """
    if not tubulin_present(pf, h):
        return False
    return int(MT_lattice[pf, h, 0]) == 1

def right_bond_count(pf: int, h: int) -> int:
    """
    Stored right-lateral-bond count for the tubulin at (pf, h).
    Returns 0 if the tubulin is absent.
    For PF0-11, this should typically be 0 or 1.
    For PF12, this may be 0, 1, or 2 because of seam bookkeeping.
    """
    if not tubulin_present(pf, h):
        return 0
    return int(MT_lattice[pf, h, 1])


def has_right_bond(pf: int, h: int) -> bool:
    """
    True if tubulin at (pf, h) has at least one right-side lateral bond.
    """
    return right_bond_count(pf, h) > 0


def has_left_bond(pf: int, h: int) -> bool:
    """
    True if tubulin at (pf, h) is laterally bonded to the protofilament on its left.

    Because MT_lattice stores only right-bond counts directly, the left bond
    for PF p is inferred from the right-bond count of PF p-1 at the same height.
    """
    if not tubulin_present(pf, h):
        return False

    pf_l = pf_left(pf)
    if not tubulin_present(pf_l, h):
        return False

    return right_bond_count(pf_l, h) > 0


def lateral_bond_count_total(pf: int, h: int) -> int:
    """
    Total number of lateral bonds touching tubulin (pf, h).
    This is reconstructed from:
      - its own right bond(s)
      - the right bond(s) of the PF to its left
    """
    if not tubulin_present(pf, h):
        return 0

    count = 0
    if has_left_bond(pf, h):
        count += 1
    if has_right_bond(pf, h):
        count += 1

    return count

def get_pocket_site_type(g: int, h: int) -> int:
    """
    Current stored site type at groove g, height h.
    """
    return int(prot_sites[g, h, 0])


def get_pocket_is_bound(g: int, h: int) -> bool:
    """
    True if an EB1 molecule is currently bound at groove g, height h.
    """
    return int(prot_sites[g, h, 2]) == 1


def get_pocket_is_bindable(g: int, h: int) -> bool:
    """
    True if a pocket is eligible for EB1 binding based on current stored state:
      - not a seam groove
      - not a SITE_SINGLE pocket
      - not already occupied
    """
    site = get_pocket_site_type(g, h)

    if groove_is_seam(g):
        return False
    if site in (SITE_SEAM, SITE_SINGLE):
        return False
    if get_pocket_is_bound(g, h):
        return False

    return True

def tubulin_can_be_removed(pf: int, h: int) -> bool:
    """
    True if tubulin (pf, h) is eligible for removal as a tip-loss event.

    Rules:
      - tubulin must exist
      - must be above the seed
      - must be the topmost tubulin on that PF
      - must have no lateral bonds
    """
    if not tubulin_present(pf, h):
        return False

    if h != (int(pf_len[pf]) - 1): # only the tip tubulin can dissociate
        return False

    if h < seed_length:
        return False

    if h <= highest_lat[pf]:
        return False

    if lateral_bond_count_total(pf, h) != 0:
        return False

    return True

def get_koff_tub(pf: int, h: int, koff_GTP: float, koff_GDP: float) -> float:
    """
    Returns the tubulin off-rate for the tip at height h on PF pf.
    Rate depends on the nucleotide state of the dimer directly below (h-1).
    koff_GTP and koff_GDP are passed in because they are increased during catastrophe.
    """
    if tubulin_is_GDP(pf, h - 1):
        return koff_GDP
    return koff_GTP

def get_right_bond_break_rate(pf: int, h: int) -> float:
    """
    Return lateral bond break rate for the right bond of tubulin (pf, h).
    Accounts for TT/TD/DD nucleotide states, Pibreak stabilization,
    seam geometry, and lateral_breaking_fold_factor.
    
    GTP vs GDP changes the conformational state of the tubulin (stiffening tubulin), which in turn affects how well it makes lateral contacts with its neighbors.
    """
    if not (tubulin_present(pf, h) and has_right_bond(pf, h)):
        return 0.0

    SEAM_PF = n_pf - 1  # PF12

    if pf == SEAM_PF:
        # Seam bond: PF12[h] connects to PF0 at h (lower) and h+1 (upper)
        # due to the 1.5-dimer offset. Which partner depends on bond count.
        gdp_self = tubulin_is_GDP(SEAM_PF, h)
        bond_count = right_bond_count(SEAM_PF, h)
        if bond_count == 1:
            # Partial bond: lower contact only, partner is PF0 at h+1
            gdp_partner = tubulin_is_GDP(0, h+1)
        else:
            # Full double bond: first bond to break is PF0 at h+2
            gdp_partner = tubulin_is_GDP(0, h+2)

        n_gdp = int(gdp_self) + int(gdp_partner)
        stabilized = has_left_bond(SEAM_PF, h)

        if n_gdp == 0:
            break_rate = k_lateralbreakSeam_TT
        elif n_gdp == 1:
            break_rate = k_lateralbreakSeam_TD
        else:
            break_rate = k_lateralbreakSeam_DD

    else:
        # PF pf connects to pf_right(pf) at same height
        gdp_self    = tubulin_is_GDP(pf, h)
        gdp_partner = tubulin_is_GDP(pf_right(pf), h)

        n_gdp = int(gdp_self) + int(gdp_partner)

        if pf == 0:
            # PF0[h] contacts PF12[h-1] and PF12[h-2]
            stabilized = (right_bond_count(SEAM_PF, h - 1) >= 1 and
                          right_bond_count(SEAM_PF, h - 2) == 2)
        else:
            stabilized = has_left_bond(pf, h) # Only need to check left bond because in the beginning we checked for right bond.
            
        if n_gdp == 0:
            break_rate = k_lateralbreak_TT
        elif n_gdp == 1:
            break_rate = k_lateralbreak_TD
        elif n_gdp == 2:
            break_rate = k_lateralbreak_DD

    if stabilized:
        break_rate /= lateral_stabilization_factor

    return break_rate * lateral_breaking_fold_factor

# =============================================================
# 
# =============================================================

def compute_pocket_nuc_state(g: int, h: int) -> int:
    """
    Computes pocket nucleotide state from MT_lattice.
    Used when updating prot_sites[g, h, 1] after a hydrolysis or tubulin event.
    
    Groove g sits between PF g-1 (left) and PF g (right).
    The nucleotide state is determined by the lower tubulins at height h
    on each side (these are the tubulins forming the base of the pocket).
    
    Returns NUC_GTP if both are GTP, NUC_GDP if both are GDP,
    NUC_MIXED if one of each.
    """
    left_gdp  = tubulin_is_GDP(g - 1, h)
    right_gdp = tubulin_is_GDP(g,     h)
    
    n_gdp = int(left_gdp) + int(right_gdp)
    
    if n_gdp == 0:
        return NUC_GTP
    elif n_gdp == 1:
        return NUC_MIXED
    else:
        return NUC_GDP
    
# =============================================================
# PROTEIN BINDING/UNBINDING HELPERS
# =============================================================

def bind_protein(g: int, h: int) -> None:
    """Mark a pocket as occupied and record the binding event."""
    prot_sites[g, h, 2] = 1
    evt_idx = len(prot_events)
    bound_prots[(g, h)] = evt_idx
    site = int(prot_sites[g, h, 0])
    nuc  = int(prot_sites[g, h, 1])
    prot_events.append({
        'g_idx':         g,
        'h':             h,
        'site_type_on':  site,
        'nuc_state_on':  nuc,
        't_on':          time_elapsed,
        't_off':         None,
        'removal':       None,
        'site_type_now': site,
        'nuc_state_now': nuc,
    })
    
    n_bindable_sites[site] -= 1     # Update count of bindable sites by site type.
    n_bound_prots_by_nuc[site][nuc] += 1   # Update count of bound proteins by site type and nucleotide state.
    
def unbind_protein(g: int, h: int, removal_reason: str) -> None:
    # NEW: break all inter-protein bonds before unbinding.
    # Handles both kinetic removal and forced site_loss removal.
    for (g2, h2) in list(protein_bonds.get((g, h), set())):
        break_protein_bond(g, h, g2, h2)

    # ...rest unchanged...
    
def unbind_protein(g: int, h: int, removal_reason: str) -> None:
    """All inter-protein bonds must be broken before unbinding.
    Mark a pocket as vacant and finalize the event record."""
    for (g2, h2) in list(protein_bonds.get((g, h), set())): #Break bonds with partners before unbinding if unbound due to tubulin loss.
        break_protein_bond(g, h, g2, h2)
    prot_sites[g, h, 2] = 0
    evt_idx = bound_prots.pop((g, h))
    prot_events[evt_idx]['t_off']   = time_elapsed
    prot_events[evt_idx]['removal'] = removal_reason
    
    site = int(prot_sites[g, h, 0])
    nuc  = int(prot_sites[g, h, 1])
    
    n_bindable_sites[site] += 1     # Update count of bindable sites by site type.
    n_bound_prots_by_nuc[site][nuc] -= 1   # Update count of bound proteins by site type and nucleotide state.


def get_possible_protein_neighbors(g: int, h: int) -> list:
    """
    All valid (g', h') within the interaction kernel of (g, h).
    Groove varies by ±1. Height varies by 0 or ±1 depending on interaction_range.
    Seam grooves (0 and n_pf) and out-of-bounds heights excluded.
    """
    neighbors = []
    for delta_g in (-1, 1):
        g2 = g + delta_g
        if g2 <= 0 or g2 >= n_pf:
            continue
        for delta_h in range(-interaction_range, interaction_range + 1):
            h2 = h + delta_h
            if not height_in_bounds(h2):
                continue
            neighbors.append((g2, h2))
    return neighbors


def prot_bond_count(g: int, h: int) -> int:
    """Number of inter-protein bonds the protein at (g, h) currently has."""
    return len(protein_bonds.get((g, h), set()))


def prot_is_doubly_bonded(g: int, h: int) -> bool:
    """
    True if bonded to at least one protein at groove g-1 AND one at groove g+1.
    Mirrors lateral_stabilization_factor logic for tubulin.
    """
    bonds = protein_bonds.get((g, h), set())
    has_left  = any(g2 == g - 1 for (g2, _) in bonds)
    has_right = any(g2 == g + 1 for (g2, _) in bonds)
    return has_left and has_right

def already_bonded_on_side(g, h, g2):
    """True if (g,h) already has a bond towards groove g2's side."""
    return any(g_b == g2 for (g_b, _) in protein_bonds.get((g, h), set()))

def get_prot_bond_break_rate(g1: int, h1: int, g2: int, h2: int) -> float:
    """
    Break rate for the bond between (g1, h1) and (g2, h2).
    Nucleotide-independent. Divided by prot_bond_stabilization_factor
    if either protein is currently doubly bonded.
    """
    rate = k_prot_bond_break
    if prot_is_doubly_bonded(g1, h1) or prot_is_doubly_bonded(g2, h2):
        rate /= prot_bond_stabilization_factor
    return rate


def form_protein_bond(g1: int, h1: int, g2: int, h2: int) -> None:
    """
    Form an inter-protein bond. Updates protein_bonds and n_bonded_prots_by_nuc.
    """
    was_unbonded_1 = (g1, h1) not in protein_bonds
    was_unbonded_2 = (g2, h2) not in protein_bonds

    protein_bonds.setdefault((g1, h1), set()) # If the key is not in the dict, inserts it with default as its value. If the key already exists, does nothing.
    protein_bonds[(g1, h1)].add((g2, h2))
    protein_bonds.setdefault((g2, h2), set())
    protein_bonds[(g2, h2)].add((g1, h1))

    # Update n_bonded_prots_by_nuc
    if was_unbonded_1:
        site = int(prot_sites[g1, h1, 0])
        nuc  = int(prot_sites[g1, h1, 1])
        if site in n_bonded_prots_by_nuc:
            n_bonded_prots_by_nuc[site][nuc] += 1

    if was_unbonded_2:
        site = int(prot_sites[g2, h2, 0])
        nuc  = int(prot_sites[g2, h2, 1])
        if site in n_bonded_prots_by_nuc:
            n_bonded_prots_by_nuc[site][nuc] += 1


def break_protein_bond(g1: int, h1: int, g2: int, h2: int) -> None:
    """
    Break an inter-protein bond. Updates protein_bonds and n_bonded_prots_by_nuc.
    """
    if (g1, h1) in protein_bonds:
        protein_bonds[(g1, h1)].discard((g2, h2))
        if not protein_bonds[(g1, h1)]: # An empty set is False
            del protein_bonds[(g1, h1)]
            site = int(prot_sites[g1, h1, 0])
            nuc  = int(prot_sites[g1, h1, 1])
            if site in n_bonded_prots_by_nuc:
                n_bonded_prots_by_nuc[site][nuc] -= 1

    if (g2, h2) in protein_bonds:
        protein_bonds[(g2, h2)].discard((g1, h1))
        if not protein_bonds[(g2, h2)]:
            del protein_bonds[(g2, h2)]
            site = int(prot_sites[g2, h2, 0])
            nuc  = int(prot_sites[g2, h2, 1])
            if site in n_bonded_prots_by_nuc:
                n_bonded_prots_by_nuc[site][nuc] -= 1
                
# =============================================================
# HELPERS TO UPDATE STATES
# =============================================================

FORCED_UNBIND_SITE_TYPES = {SITE_EMPTY, SITE_SINGLE}

def update_pocket(g: int, h: int) -> None:
    old_site_type = int(prot_sites[g, h, 0])
    old_nuc_state = int(prot_sites[g, h, 1])
    
    new_site_type = classify_pocket(g, h, pf_len)
    new_nuc_state = compute_pocket_nuc_state(g, h)
    
    is_bound = get_pocket_is_bound(g, h)
        
    if is_bound:
        if new_site_type in FORCED_UNBIND_SITE_TYPES:
            unbind_protein(g, h, removal_reason="site_loss")
            
            # unbind_protein() already added +1 to the old unoccupied site type, so we need to move it to the new site type.
            if old_site_type in n_bindable_sites:
                n_bindable_sites[old_site_type] -= 1
            if new_site_type in n_bindable_sites:
                n_bindable_sites[new_site_type] += 1
            
            prot_sites[g, h, 0] = new_site_type
            prot_sites[g, h, 1] = new_nuc_state
            return
        
        if (old_site_type != new_site_type) or (old_nuc_state != new_nuc_state):
            # Change the dictionary count
            if old_site_type in n_bound_prots_by_nuc:
                n_bound_prots_by_nuc[old_site_type][old_nuc_state] -= 1
            if new_site_type in n_bound_prots_by_nuc:
                n_bound_prots_by_nuc[new_site_type][new_nuc_state] += 1

    else:
        if old_site_type != new_site_type:
            # Change the dictionary count
            if old_site_type in n_bindable_sites:
                n_bindable_sites[old_site_type] -= 1
            if new_site_type in n_bindable_sites:
                n_bindable_sites[new_site_type] += 1

    prot_sites[g, h, 0] = new_site_type
    prot_sites[g, h, 1] = new_nuc_state


def update_prot_events_now(g: int, h: int) -> None:
    if (g, h) not in bound_prots:
        return
    evt_idx = bound_prots[(g, h)]
    prot_events[evt_idx]["site_type_now"] = int(prot_sites[g, h, 0])
    prot_events[evt_idx]["nuc_state_now"] = int(prot_sites[g, h, 1])
    

def refresh_all(g: int, h: int) -> None:
    if g <= 0 or g >= n_pf:
        return
    if h < 0 or not height_in_bounds(h):
        return
    update_pocket(g, h) #Recompute the pocket itself
    update_prot_events_now(g, h) #If something was bound there, update prot_events annotations to match the new pocket state.

def refresh_local_environment_after_tubulin_change(pf: int, h: int) -> None:
    """
    Recompute only the pockets affected by a change to tubulin (pf, h).

    A tubulin at (pf, h) can affect:
      - groove pf     : between PF(pf-1) and PF(pf)
      - groove pf + 1 : between PF(pf)   and PF(pf+1)
      - pocket rows h and h-1
    """
    grooves = (pf, pf + 1)
    heights = (h - 1, h)

    for g in grooves:
        if 0 <= g <= n_pf:
            for hh in heights:
                refresh_all(g, hh)
    
# =============================================================
# 
# =============================================================
 
def height_in_bounds(h: int) -> bool:
    """True if height h is valid for the preallocated arrays."""
    return 0 <= h < MT_lattice.shape[1]

def require_height_in_bounds(h: int) -> None:
    """
    Raise an error if h is outside the allocated simulation height.
    """
    if not height_in_bounds(h):
        raise RuntimeError(
            f"Simulation exceeded preallocated height capacity: h={h}, "
            f"max_valid={MT_lattice.shape[1] - 1}"
        )