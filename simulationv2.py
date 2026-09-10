# =============================================================
# simulation.py
# =============================================================
# Main Gillespie simulation loop for the microtubule + EB1-like model.
# This file imports the initialized state from initialization.py.

import math
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
from PIL import Image
from datetime import datetime

from params import *
from helpers import classify_pocket
from initialization import *
import initialization as st
from simulation_helpers import *

def sample_exponential_dt(rate: float) -> float:
    """
    Sample a waiting time from an exponential distribution with parameter 'rate'.
    dt = -ln(u) / rate, where u ~ Uniform(0,1).
    """
    if rate <= 0:
        return np.inf

    u = np.random.random()
    while u == 0.0:
        u = np.random.random()
    return -math.log(u) / rate

def make_event(event_type: str, rate: float, h: int, pf: int = None, g: int = None) -> dict:
    """
      event_type : string label like 'tub_add', 'eb_bind', etc.
      rate       : event rate
      h          : height index
    Optional:
      pf         : protofilament index
      g          : groove index
    """
    event = {
        "event_type": event_type,
        "rate": float(rate),
        "dt": sample_exponential_dt(rate),
        "h": int(h),
        "pf": None if pf is None else int(pf),
        "g": None if g is None else int(g),
    }
    return event

def choose_next_event(candidate_events):
    """
    Return the candidate event with the smallest sampled dt.
    """
    possible = [ev for ev in candidate_events]
    if not possible:
        return None
    return min(possible, key=lambda ev: ev["dt"])

# -----------------------------------------------------------
# GTP HYDROLYSIS
# -----------------------------------------------------------

def update_highest_full_GDP(hydrolyzed_hh: int) -> None:
    """
    Walk up from st.highest_full_GDP checking if all PFs are GDP at each row.
    Stop at the first row where any PF is still GTP.
    up_to: the height of the dimer that just hydrolyzed.
    """
    for hh in range(st.highest_full_GDP + 1, hydrolyzed_hh + 1):
        if all(st.MT_lattice[pf, hh, 0] == 1 for pf in range(n_pf)):
            st.highest_full_GDP = hh
        else:
            break

def execute_hydrolysis(dt: float) -> None:
    """
    Only buried dimers hydrolyze, so the loop to sample a waiting time goes backwards from the second to last dimer of each PF.
    If that waiting time < dt, the dimer hydrolyzes.
    Hydrolysis check called once per iteration after the winning event fires. This is computationally more efficient than putting all hydrolysis events in the candidate list for Guillespie.
    """
    for pf in range(n_pf):
        for hh in range(pf_len[pf] - 2, st.highest_full_GDP, -1):

            if st.MT_lattice[pf, hh, 0] == 1:  # already GDP
                continue

            if sample_exponential_dt(k_hydrolysis) < dt:
                st.MT_lattice[pf, hh, 0] = 1   # GTP → GDP
                update_highest_full_GDP(hydrolyzed_hh=hh)
                refresh_local_environment_after_tubulin_change(pf, hh)
                
# -----------------------------------------------------------
# EVENT GENERATION
# -----------------------------------------------------------

def generate_tub_add_events():
    candidate_events = []
    for pf_idx in range(n_pf):
        hh = pf_len[pf_idx]  # next available height where a new tubulin can go
        event = make_event('tub_add', rate=kon_tub * conc_tubGTP, h=hh, pf=pf_idx)
        candidate_events.append(event)
    return candidate_events

def generate_tub_removal_events():
    candidate_events = []
    for pf_idx in range(n_pf):
        hh = pf_len[pf_idx] - 1  # last tubulin
        if not tubulin_can_be_removed(pf_idx, hh):
            continue
        if not height_in_bounds(hh):
            continue
        event = make_event('tub_removal', rate=get_koff_tub(pf_idx, hh, koff_tubGTP, koff_tubGDP), h=hh, pf=pf_idx)
        candidate_events.append(event)
    return candidate_events

def generate_lat_bond_formation_events():
    candidate_events = []
    for pf_idx in range(n_pf):
        hh     = highest_lat[pf_idx]      # highest fully bonded subunit
        new_hh = hh + 1                   # next candidate for lateral bond formation

        if new_hh >= pf_len[pf_idx]:      # no free tubulin waiting — normal, skip
            continue

        if pf_idx == n_pf - 1:  # PF12 seam
            bond_count = right_bond_count(pf_idx, new_hh)
            if bond_count == 2:  # already fully bonded
                continue
            # check PF0 is long enough for the seam contact
            if bond_count == 1 and new_hh >= pf_len[0] - 2:
                continue
            if bond_count == 0 and new_hh >= pf_len[0] - 1:
                continue
            rate = k_lateralbondSeam
        else:
            if not tubulin_present(pf_right(pf_idx), new_hh):  # right neighbor must exist
                continue
            rate = k_lateralbond

        event = make_event('lat_bond_form', rate=rate, h=new_hh, pf=pf_idx)
        candidate_events.append(event)
    return candidate_events

def generate_lat_bond_break_events():
    candidate_events = []
    for pf_idx in range(n_pf):
        # only break bonds above the seed
        hh = highest_lat[pf_idx]
        if right_bond_count(pf_idx, hh) == 0:
            continue
        if hh < seed_length:
            continue
        
        rate = get_right_bond_break_rate(pf_idx, hh)
        if rate > 0:
            event = make_event('lat_bond_break', rate=rate, h=hh, pf=pf_idx)
            candidate_events.append(event)
    return candidate_events

def generate_prot_bind_events():
    """
    One event for all edge-bindable pockets, one for all lattice-bindable pockets. So events are grouped and time is calculated for all.
    Rate = kon_per_site * conc_nM * n_sites.
    Follows MATLAB: EdgeProtP = -log(rand) / (kOnProtEdge * n_edge_sites).
    """
    candidate_events = []
    conc_nM = conc_prot * 1e3

    n_edge    = (n_bindable_sites[SITE_EDGELAT2] +
                 n_bindable_sites[SITE_EDGELONG2] +
                 n_bindable_sites[SITE_EDGE3])
    n_lattice =  n_bindable_sites[SITE_LATTICE]

    if n_edge > 0:
        rate = kon_prot_1 * conc_nM * n_edge
        candidate_events.append(make_event('prot_bind_edge', rate=rate, h=0))

    if n_lattice > 0:
        rate = kon_prot_0 * conc_nM * n_lattice
        candidate_events.append(make_event('prot_bind_lattice', rate=rate, h=0))

    return candidate_events

def generate_prot_remove_events():
    candidate_events = []

    n_gtp_edge = n_gdp_edge = n_gtp_lattice = n_gdp_lattice = 0

    for (g, h) in bound_prots:
        # Bonded proteins are no longer skipped. Everything lets go at koff;
        # what differs is whether it leaves or stays tethered, which
        # execute_prot_remove decides.
        site = int(st.prot_sites[g, h, 0])
        nuc  = int(st.prot_sites[g, h, 1])
        is_lattice = (site == SITE_LATTICE)
        is_gdp     = (nuc != NUC_GTP)

        if     is_lattice and not is_gdp: n_gtp_lattice += 1
        elif   is_lattice and     is_gdp: n_gdp_lattice += 1
        elif not is_lattice and not is_gdp: n_gtp_edge   += 1
        elif not is_lattice and     is_gdp: n_gdp_edge   += 1

    if n_gtp_edge     > 0:
        candidate_events.append(make_event('prot_remove_gtp_edge',    rate=koff_prot_GTP_1 * n_gtp_edge,    h=-1))
    if n_gdp_edge     > 0:
        candidate_events.append(make_event('prot_remove_gdp_edge',    rate=koff_prot_GDP_1 * n_gdp_edge,    h=-1))
    if n_gtp_lattice  > 0:
        candidate_events.append(make_event('prot_remove_gtp_lattice', rate=koff_prot_GTP_0 * n_gtp_lattice, h=-1))
    if n_gdp_lattice  > 0:
        candidate_events.append(make_event('prot_remove_gdp_lattice', rate=koff_prot_GDP_0 * n_gdp_lattice, h=-1))

    return candidate_events

def generate_prot_reattach_events():
    """
    A tethered subunit coming back onto the lattice.

    Rate is kon for the target pocket times conc_prot_tethered_nM -- the LOCAL
    concentration it sees while the ring holds it, not bulk conc_prot. Grouped
    by target pocket type the same way the binding events are.
    """
    if conc_prot_tethered_nM <= 0.0 or not tethered_prots:
        return []

    counts = {}
    for (g, h) in tethered_prots:
        for h_new in tether_reattach_rows(g, h):
            site = int(st.prot_sites[g, h_new, 0])
            nuc = int(st.prot_sites[g, h_new, 1])
            key = ('lattice' if site == SITE_LATTICE else 'edge', nuc != NUC_GTP)
            counts[key] = counts.get(key, 0) + 1

    KON = {'lattice': kon_prot_0, 'edge': kon_prot_1}
    candidate_events = []
    for (cls, gdp), n in counts.items():
        name = f"prot_reattach_{'gdp' if gdp else 'gtp'}_{cls}"
        rate = KON[cls] * conc_prot_tethered_nM * n
        candidate_events.append(make_event(name, rate=rate, h=-1))
    return candidate_events

def generate_prot_bond_form_events():
    """
    One event per unbonded pair of neighboring bound proteins.
    g < g2 filter avoids generating each pair twice.
    """
    candidate_events = []
    for (g, h) in bound_prots:
        for (g2, h2) in get_possible_protein_neighbors(g, h):
            if g2 <= g: # g2 <= g filter avoids generating each pair twice.
                continue
            if already_bonded_on_side(g, h, g2):
                continue
            if already_bonded_on_side(g2, h2, g):
                continue
            if not get_pocket_is_bound(g2, h2):
                continue
            if (g2, h2) in protein_bonds.get((g, h), set()): # Already bonded with each other
                continue
            if not oligomer_span_ok(g, h, g2, h2):
                continue   # merged oligomer would exceed 2*interaction_range rows
            rate_bond = k_prot_bond_form_same_h if h == h2 else k_prot_bond_form_diff_h
            event = make_event('prot_bond_form', rate=rate_bond, h=h, g=g)
            event['g2'] = g2
            event['h2'] = h2
            candidate_events.append(event)
    return candidate_events

def generate_prot_bond_break_events():
    """
    One event per existing inter-protein bond.
    g < g2 filter avoids generating each bond twice.
    """
    candidate_events = []
    for (g, h), neighbors in protein_bonds.items():
        for (g2, h2) in neighbors:
            if g2 <= g: # g2 <= g filter avoids generating each pair twice.
                continue
            rate = get_prot_bond_break_rate(g, h, g2, h2)
            event = make_event('prot_bond_break', rate=rate, h=h, g=g)
            event['g2'] = g2
            event['h2'] = h2
            candidate_events.append(event)
    return candidate_events
    
# -----------------------------------------------------------
# EVENT EXECUTION
# -----------------------------------------------------------

def execute_tub_add(event: dict) -> None:
    pf = event['pf']
    h  = event['h']

    if int(pf_len[pf]) != h: # sanity check
        raise RuntimeError(f"tub_add height mismatch: pf={pf}, h={h}, pf_len={pf_len[pf]}")

    # Make room before writing. +2 of headroom covers the seam lookups in
    # get_right_bond_break_rate, which reach up to h+2 on PF0.
    st.ensure_height(h + 2)

    pf_len[pf] += 1 # extend the protofilament

    # new dimer is GTP by default (st.MT_lattice already zero-initialized)
    # st.MT_lattice[pf, h, 0] = 0  # explicit but redundant for fresh positions

    refresh_local_environment_after_tubulin_change(pf, h) # recompute affected pockets

def execute_tub_remove(event: dict) -> None:
    pf = event["pf"]
    h = event["h"]

    # safety checks
    if not tubulin_present(pf, h):
        raise RuntimeError(f"Tried to remove absent tubulin at PF {pf}, h={h}")

    if h != (pf_len[pf]-1):
        raise RuntimeError(f"Tried to remove non-tip tubulin at PF {pf}, h={h}")

    if not tubulin_can_be_removed(pf, h):
        raise RuntimeError(f"Tubulin at PF {pf}, h={h} is not removable")

    # clear stored state at that lattice site (optional but cleaner)
    st.MT_lattice[pf, h, 0] = 0   # hydrolysis layer reset
    st.MT_lattice[pf, h, 1] = 0   # right-bond count reset

    pf_len[pf] -= 1 # shorten PF
    refresh_local_environment_after_tubulin_change(pf, h)
    
def execute_lat_bond_form(event: dict) -> None:
    pf = event['pf']
    h  = event['h']

    if not tubulin_present(pf, h):
        raise RuntimeError(f"lat_bond_form: no tubulin at PF {pf}, h={h}")

    SEAM_PF = n_pf - 1
    if pf == SEAM_PF:
        current = right_bond_count(pf, h)
        if current >= 2:
            raise RuntimeError(f"lat_bond_form seam: already fully bonded at PF {pf}, h={h}")
        st.MT_lattice[pf, h, 1] += 1
        if st.MT_lattice[pf, h, 1] == 2:   # seam needs both bonds to be "fully bonded"
            highest_lat[pf] = h
    else:
        if right_bond_count(pf, h) >= 1:
            raise RuntimeError(f"lat_bond_form: already bonded at PF {pf}, h={h}")
        st.MT_lattice[pf, h, 1] = 1
        highest_lat[pf] = h              # h is now the new highest bonded

    refresh_local_environment_after_tubulin_change(pf, h)


def execute_lat_bond_break(event: dict) -> None:
    pf = event['pf']
    h  = event['h']

    if right_bond_count(pf, h) == 0:
        raise RuntimeError(f"lat_bond_break: no right bond at PF {pf}, h={h}")

    st.MT_lattice[pf, h, 1] -= 1

    SEAM_PF = n_pf - 1
    if pf == SEAM_PF:
        # seam: fully bonded = count 2. Going 2→1 means no longer fully bonded.
        if h == highest_lat[pf]:
            highest_lat[pf] = h - 1
    else:
        if h <= highest_lat[pf]:
            highest_lat[pf] = h - 1

    refresh_local_environment_after_tubulin_change(pf, h)

def execute_prot_bind(event: dict) -> None:
    EDGE_SITE_TYPES = (SITE_EDGELAT2, SITE_EDGELONG2, SITE_EDGE3)
    lattice_event = 'lattice' in event['event_type']

    pockets = []
    max_h = int(pf_len.max()) # No pocket above the tallest PF can exist, so no need to check above that height
    for g in range(1, n_pf):
        for h in range(max_h):
            if not get_pocket_is_bindable(g, h):
                continue
            site = get_pocket_site_type(g, h)
            if lattice_event and site == SITE_LATTICE:
                pockets.append((g, h))
            elif not lattice_event and site in EDGE_SITE_TYPES:
                pockets.append((g, h))

    if not pockets:
        raise RuntimeError(f"execute_prot_bind: no pockets available for event type '{event['event_type']}' — counter mismatch")

    g, h = pockets[np.random.randint(len(pockets))]
    bind_protein(g, h)
    
def execute_prot_remove(event: dict) -> None:

    EDGE_SITE_TYPES = (SITE_EDGELAT2, SITE_EDGELONG2, SITE_EDGE3)
    # The types of events are 'prot_remove_gtp_edge', 'prot_remove_gdp_edge', 'prot_remove_gtp_lattice', 'prot_remove_gdp_lattice'
    lattice_event = 'lattice' in event['event_type']
    gdp_event     = 'gdp'     in event['event_type']

    proteins = [] # Create the pool of candidate proteins to remove
    for (g, h) in bound_prots:
        site = int(st.prot_sites[g, h, 0])
        nuc  = int(st.prot_sites[g, h, 1])
        is_lattice = (site == SITE_LATTICE)
        is_gdp     = (nuc != NUC_GTP)
        if is_lattice == lattice_event and is_gdp == gdp_event:
            proteins.append((g, h))

    if not proteins:
        raise RuntimeError(f"execute_prot_remove: pool '{event['event_type']}' is empty at execution — counter mismatch")

    g, h = proteins[np.random.randint(len(proteins))]

    if prot_bond_count(g, h) == 0 or conc_prot_tethered_nM <= 0.0:
        unbind_protein(g, h, event['event_type'])   # nothing holds it: gone
    else:
        detach_to_tether(g, h)                      # the oligomer keeps it
    
    
def execute_prot_reattach(event: dict) -> None:
    """Pick one (tethered subunit, target row) pair of this class and land it."""
    lattice_event = 'lattice' in event['event_type']
    gdp_event = 'gdp' in event['event_type']

    pool = []
    for (g, h) in tethered_prots:
        for h_new in tether_reattach_rows(g, h):
            site = int(st.prot_sites[g, h_new, 0])
            nuc = int(st.prot_sites[g, h_new, 1])
            if (site == SITE_LATTICE) != lattice_event:
                continue
            if (nuc != NUC_GTP) != gdp_event:
                continue
            pool.append((g, h, h_new))

    if not pool:
        raise RuntimeError(f"execute_prot_reattach: pool '{event['event_type']}' is "
                           f"empty at execution - counter mismatch")

    g, h, h_new = pool[np.random.randint(len(pool))]
    reattach_from_tether(g, h, h_new)

def execute_prot_bond_form(event: dict) -> None:
    g1, h1, g2, h2 = event['g'], event['h'], event['g2'], event['h2']
    if not get_pocket_is_bound(g1, h1) or not get_pocket_is_bound(g2, h2):
        raise RuntimeError(f"execute_prot_bond_form: one of the pockets is not bound at execution — counter mismatch: (g1,h1)=({g1},{h1}), (g2,h2)=({g2},{h2})")
    if (g2, h2) in protein_bonds.get((g1, h1), set()):
        raise RuntimeError(f"execute_prot_bond_form: pockets already bonded at execution — counter mismatch: (g1,h1)=({g1},{h1}), (g2,h2)=({g2},{h2})")
    if not oligomer_span_ok(g1, h1, g2, h2):
        raise RuntimeError(f"execute_prot_bond_form: merged oligomer would span more than "
                           f"{2 * interaction_range} rows: (g1,h1)=({g1},{h1}), (g2,h2)=({g2},{h2})")
    form_protein_bond(g1, h1, g2, h2)

def execute_prot_bond_break(event: dict) -> None:
    g1, h1, g2, h2 = event['g'], event['h'], event['g2'], event['h2']
    if (g2, h2) not in protein_bonds.get((g1, h1), set()):
        raise RuntimeError(f"execute_prot_bond_break: pockets not bonded at execution — counter mismatch: (g1,h1)=({g1},{h1}), (g2,h2)=({g2},{h2})")
    break_protein_bond(g1, h1, g2, h2)
    
# -----------------------------------------------------------
# PRINTERS
# -----------------------------------------------------------

frames = []  # collects one PIL image per snapshot

def plot_pf_lengths_and_lattice_occupancy():
    # 2-row layout: top row has tip region + full lattice, bottom row has protein bonds at tip
    fig = plt.figure(figsize=(14, 16))
    ax1 = fig.add_subplot(2, 2, 1)  # top-left:  tip region
    ax2 = fig.add_subplot(2, 2, 2)  # top-right: full lattice
    ax3 = fig.add_subplot(2, 1, 2)  # bottom:    protein bonds at tip (full width)

    max_h    = int(pf_len.max())
    tip_window = 80
    tip_start  = max(0, max_h - tip_window)

    cmap = matplotlib.colors.ListedColormap(['white', '#0a2472', '#4a90d9'])
    legend_elements = [
        Patch(facecolor='#0a2472', label='GTP'),
        Patch(facecolor='#4a90d9', label='GDP'),
    ]

    # --- ax1: tip region ---
    tip_img = np.zeros((max_h - tip_start, n_pf), dtype=int)
    for pf in range(n_pf):
        for h in range(tip_start, pf_len[pf]):
            tip_img[h - tip_start, pf] = 1 if st.MT_lattice[pf, h, 0] == 0 else 2

    ax1.imshow(tip_img, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax1.set_xlabel("Protofilament")
    ax1.set_ylabel(f"Height (rows {tip_start}–{max_h})")
    ax1.set_title(f"Tip region at t = {st.time_elapsed:.4f} s")
    ax1.set_yticks(np.arange(0, max_h - tip_start, 5))
    ax1.set_yticklabels(np.arange(tip_start, max_h, 5))
    ax1.legend(handles=legend_elements, loc='upper right')

    for pf in range(n_pf - 1):
        for h in range(max(seed_length, tip_start), pf_len[pf]):
            if right_bond_count(pf, h) > 0:
                ax1.plot([pf + 0.3, pf + 0.7], [h - tip_start, h - tip_start], color='orange', linewidth=1.5)

    for h in range(max(seed_length, tip_start), pf_len[n_pf - 1]):
        bc = right_bond_count(n_pf - 1, h)
        if bc == 1:
            ax1.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h - tip_start, h - tip_start], color='yellow', linewidth=1.5)
        elif bc == 2:
            ax1.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h - tip_start, h - tip_start], color='red', linewidth=1.5)

    for g in range(1, n_pf):
        for h in range(tip_start, max_h):
            if st.prot_sites[g, h, 2] == 1:
                ax1.plot(g - 0.5, h - tip_start, '.', color='#00FF00', markersize=8)

    # --- ax2: full lattice ---
    lattice_img = np.zeros((max_h, n_pf), dtype=int)
    for pf in range(n_pf):
        for h in range(pf_len[pf]):
            lattice_img[h, pf] = 1 if st.MT_lattice[pf, h, 0] == 0 else 2

    ax2.imshow(lattice_img, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax2.set_xlabel("Protofilament")
    ax2.set_ylabel("Height")
    ax2.set_title(f"Full lattice at t = {st.time_elapsed:.4f} s")
    ax2.legend(handles=legend_elements, loc='upper right')

    for g in range(1, n_pf):
        for h in range(max_h):
            if st.prot_sites[g, h, 2] == 1:
                ax2.plot(g - 0.5, h, '.', color='#00FF00', markersize=8)

    for pf in range(n_pf - 1):
        for h in range(seed_length, pf_len[pf]):
            if right_bond_count(pf, h) > 0:
                ax2.plot([pf + 0.3, pf + 0.7], [h, h], color='orange', linewidth=1.5)

    for h in range(seed_length, pf_len[n_pf - 1]):
        bc = right_bond_count(n_pf - 1, h)
        if bc == 1:
            ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='yellow', linewidth=1.5)
        elif bc == 2:
            ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='red', linewidth=1.5)

    # --- ax3: protein bonds at tip region ---
    tip_img3 = np.zeros((max_h - tip_start, n_pf), dtype=int)
    for pf in range(n_pf):
        for h in range(tip_start, pf_len[pf]):
            tip_img3[h - tip_start, pf] = 1 if st.MT_lattice[pf, h, 0] == 0 else 2

    ax3.imshow(tip_img3, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax3.set_xlabel("Protofilament")
    ax3.set_ylabel(f"Height (rows {tip_start}–{max_h})")
    ax3.set_title(f"Inter-protein bonds at tip, t = {st.time_elapsed:.4f} s  "
                  f"({len(protein_bonds)} bonded proteins, "
                  f"{sum(len(v) for v in protein_bonds.values()) // 2} bonds)")
    ax3.set_yticks(np.arange(0, max_h - tip_start, 5))
    ax3.set_yticklabels(np.arange(tip_start, max_h, 5))

    # draw all bound proteins in tip region
    for g in range(1, n_pf):
        for h in range(tip_start, max_h):
            if st.prot_sites[g, h, 2] == 1:
                color = '#00CC00' if (g, h) in protein_bonds else '#9ACD32'
                ax3.plot(g - 0.5, h - tip_start, 'o', color=color, markersize=7, zorder=3)

    # draw inter-protein bond lines (only if both proteins are in tip window)
    drawn_bonds = set()
    for (g1, h1), neighbors in protein_bonds.items():
        if g1 < 1 or g1 >= n_pf or h1 < tip_start:
            continue
        for (g2, h2) in neighbors:
            if g2 < 1 or g2 >= n_pf or h2 < tip_start:
                continue
            bond_key = (min(g1, g2), min(h1, h2), max(g1, g2), max(h1, h2))
            if bond_key in drawn_bonds:
                continue
            drawn_bonds.add(bond_key)
            ax3.plot([g1 - 0.5, g2 - 0.5], [h1 - tip_start, h2 - tip_start],
                     '-', color='red', linewidth=1.5, zorder=2)

    bond_legend = [
        Patch(facecolor='#0a2472', label='GTP'),
        Patch(facecolor='#4a90d9', label='GDP'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9ACD32',
                   markersize=7, label='unbound protein'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CC00',
                   markersize=7, label='bonded protein'),
        plt.Line2D([0], [0], color='red', linewidth=1.5, label='inter-protein bond'),
    ]
    ax3.legend(handles=bond_legend, loc='upper right')

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)
    
# def plot_pf_lengths_and_lattice_occupancy():
#     # 2-row layout: top row has PF lengths + lattice, bottom row has protein bonds
#     fig = plt.figure(figsize=(14, 8))
#     ax1 = fig.add_subplot(2, 2, 1)  # top-left:  PF lengths
#     ax2 = fig.add_subplot(2, 2, 2)  # top-right: lattice occupancy
#     ax3 = fig.add_subplot(2, 1, 2)  # bottom:    protein bonds (full width)

#     # --- ax1: PF lengths bar chart ---
#     ax1.bar(np.arange(n_pf), pf_len.copy(), width=1.0, edgecolor='blue', linewidth=0.3)
#     ax1.set_xlabel("Protofilament")
#     ax1.set_ylabel("Length")
#     ax1.set_title(f"PF lengths at t = {st.time_elapsed:.4f} s")

#     # --- ax2: lattice occupancy ---
#     max_h = int(pf_len.max())
#     lattice_img = np.zeros((max_h, n_pf), dtype=int)
#     for pf in range(n_pf):
#         for h in range(pf_len[pf]):
#             if st.MT_lattice[pf, h, 0] == 0:
#                 lattice_img[h, pf] = 1   # GTP
#             else:
#                 lattice_img[h, pf] = 2   # GDP

#     cmap = matplotlib.colors.ListedColormap(['white', '#0a2472', '#4a90d9'])
#     ax2.imshow(lattice_img, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
#     ax2.set_xlabel("Protofilament")
#     ax2.set_ylabel("Height")
#     ax2.set_title(f"Lattice + protein occupancy at t = {st.time_elapsed:.4f} s")

#     legend_elements = [
#         Patch(facecolor='#0a2472', label='GTP'),
#         Patch(facecolor='#4a90d9', label='GDP'),
#     ]
#     ax2.legend(handles=legend_elements, loc='upper right')

#     for g in range(1, n_pf):
#         for h in range(max_h):
#             if st.prot_sites[g, h, 2] == 1:
#                 ax2.plot(g - 0.5, h, '.', color='#00FF00', markersize=8)

#     for pf in range(n_pf - 1):
#         for h in range(seed_length, pf_len[pf]):
#             if right_bond_count(pf, h) > 0:
#                 ax2.plot([pf + 0.3, pf + 0.7], [h, h], color='orange', linewidth=1.5)

#     for h in range(seed_length, pf_len[n_pf - 1]):
#         bc = right_bond_count(n_pf - 1, h)
#         if bc == 1:
#             ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='yellow', linewidth=1.5)
#         elif bc == 2:
#             ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='red', linewidth=1.5)

#     # --- ax3: replica of ax2 with inter-protein bond overlay ---
#     groove_img2 = np.zeros((max_h, n_pf), dtype=int)
#     for pf in range(n_pf):
#         for h in range(pf_len[pf]):
#             if st.MT_lattice[pf, h, 0] == 0:
#                 groove_img2[h, pf] = 1   # GTP
#             else:
#                 groove_img2[h, pf] = 2   # GDP

#     cmap2 = matplotlib.colors.ListedColormap(['white', '#0a2472', '#4a90d9'])
#     ax3.imshow(groove_img2, origin='lower', aspect='auto', cmap=cmap2, vmin=0, vmax=2)
#     ax3.set_xlabel("Protofilament")
#     ax3.set_ylabel("Height")
#     ax3.set_title(f"Inter-protein bonds at t = {st.time_elapsed:.4f} s  "
#                   f"({len(protein_bonds)} bonded proteins, "
#                   f"{sum(len(v) for v in protein_bonds.values()) // 2} bonds)")

#     # draw all bound proteins: unbound in light yellowish-green, bonded in bright green
#     for g in range(1, n_pf):
#         for h in range(max_h):
#             if st.prot_sites[g, h, 2] == 1:
#                 color = '#00CC00' if (g, h) in protein_bonds else '#9ACD32'
#                 ax3.plot(g - 0.5, h, 'o', color=color, markersize=7, zorder=3)

#     # draw inter-protein bond lines
#     drawn_bonds = set()
#     for (g1, h1), neighbors in protein_bonds.items():
#         if g1 < 1 or g1 >= n_pf:
#             continue
#         for (g2, h2) in neighbors:
#             if g2 < 1 or g2 >= n_pf:
#                 continue
#             bond_key = (min(g1, g2), min(h1, h2), max(g1, g2), max(h1, h2))
#             if bond_key in drawn_bonds:
#                 continue
#             drawn_bonds.add(bond_key)
#             ax3.plot([g1 - 0.5, g2 - 0.5], [h1, h2], '-', color='red',
#                      linewidth=1.5, zorder=2)

#     ax3.set_xticks(np.arange(n_pf))
#     ax3.set_xticklabels(np.arange(n_pf))

#     bond_legend = [
#         Patch(facecolor='#0a2472', label='GTP'),
#         Patch(facecolor='#4a90d9', label='GDP'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9ACD32',
#                    markersize=7, label='unbound protein'),
#         plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00CC00',
#                    markersize=7, label='bonded protein'),
#         plt.Line2D([0], [0], color='red', linewidth=1.5, label='inter-protein bond'),
#     ]
#     ax3.legend(handles=bond_legend, loc='upper right')

#     fig.tight_layout()

#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight')
#     buf.seek(0)
#     frames.append(Image.open(buf).copy())
#     buf.close()
#     plt.close(fig)
    
# def plot_pf_lengths_and_lattice_occupancy():
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

#     # --- left: PF lengths bar chart (no gaps) ---
#     ax1.bar(np.arange(n_pf), pf_len.copy(), width=1.0, edgecolor='blue', linewidth=0.3)
#     ax1.set_xlabel("Protofilament")
#     ax1.set_ylabel("Length")
#     ax1.set_title(f"PF lengths at t = {st.time_elapsed:.4f} s")
    
#     # --- right: lattice occupancy with protein overlay and GTP/GDP coloring ---
#     max_h = int(pf_len.max())
#     lattice_img = np.zeros((max_h, n_pf), dtype=int)
#     for pf in range(n_pf):
#         for h in range(pf_len[pf]):
#             if st.MT_lattice[pf, h, 0] == 0:
#                 lattice_img[h, pf] = 1   # GTP
#             else:
#                 lattice_img[h, pf] = 2   # GDP

#     cmap = matplotlib.colors.ListedColormap(['white', '#0a2472', '#4a90d9'])  # 0 = background (white), 1 = GTP (dark blue), 2 = GDP (medium blue)
#     im = ax2.imshow(lattice_img, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
#     ax2.set_xlabel("Protofilament")
#     ax2.set_ylabel("Height")
#     ax2.set_title(f"Lattice + protein occupancy at t = {st.time_elapsed:.4f} s")
    
#     # fig.colorbar(im, ax=ax2, label="Tubulin present")
#     # cbar = fig.colorbar(im, ax=ax2, ticks=[1, 2])
#     # cbar.ax.set_yticklabels(['GTP', 'GDP'])
    
#     # legend_elements = [
#     #     Patch(facecolor='#4a90d9', label='GTP'),
#     #     Patch(facecolor='#0a2472', label='GDP'),
#     # ]
#     legend_elements = [
#         Patch(facecolor='#0a2472', label='GTP'),
#         Patch(facecolor='#4a90d9', label='GDP'),
#     ]
#     ax2.legend(handles=legend_elements, loc='upper right')

#     # overlay bound proteins as dots
#     for g in range(1, n_pf):  # skip seam grooves 0 and n_pf
#         for h in range(max_h):
#             if st.prot_sites[g, h, 2] == 1:
#                 ax2.plot(g - 0.5, h, '.', color='#00FF00', markersize=8)  # g-0.5 centers the dot between the two PFs

#     # overlay lateral bonds as horizontal lines between PF columns
#     # each cell in imshow is centered at x=pf, so the gap between pf and pf+1 spans x=[pf+0.5, pf+1-0.5]
#     # we draw a short line from pf+0.1 to pf+0.9 at height h
#     for pf in range(n_pf - 1):  # PF0-11: right bond connects pf to pf+1
#         for h in range(seed_length, pf_len[pf]):
#             if right_bond_count(pf, h) > 0:
#                 ax2.plot([pf + 0.3, pf + 0.7], [h, h], color='orange', linewidth=1.5)

#     # seam: PF12 bonds to PF0, drawn on the right edge of PF12 (x = n_pf - 1 + 0.5 area)
#     # since PF0 is at x=0 and PF12 is at x=12, the seam wraps; mark it on PF12's right side
#     for h in range(seed_length, pf_len[n_pf - 1]):
#         bc = right_bond_count(n_pf - 1, h)
#         if bc == 1:
#             ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='yellow', linewidth=1.5)
#         elif bc == 2:
#             ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='red', linewidth=1.5)

#     # render to in-memory buffer and store as PIL image
#     buf = io.BytesIO() # Creates a fake file that lives in RAM instead of on disk
#     fig.savefig(buf, format='png', bbox_inches='tight') # Write the figure to the RAM buffer
#     buf.seek(0)
#     frames.append(Image.open(buf).copy()) # will read image from first line thanks to buf.seek(0)
#     buf.close() # Frees the RAM used by the buffer
#     plt.close(fig)
    
def print_selected_event(event):
    """
    Print the selected winning event.
    """
    if event is None:
        print("  Selected event: None")
        return

    print(
        "  Selected event: "
        f"type={event['event_type']} "
        f"pf={event.get('pf')} "
        f"g={event.get('g')} "
        f"h={event['h']} "
        f"rate={event['rate']:.6g} "
        f"dt={event['dt']:.6g}"
    )
    
def print_candidate_events(candidates):
    """
    Print all candidate events sorted by sampled dt.
    """
    if not candidates:
        print("  No candidate events.")
        return

    candidates_sorted = sorted(candidates, key=lambda ev: ev["dt"])

    print("  Candidate events:")
    for i, ev in enumerate(candidates_sorted, start=1):
        print(
            f"    {i:>2d}. "
            f"type={ev['event_type']:<10s} "
            f"pf={str(ev.get('pf')):<4s} "
            f"g={str(ev.get('g')):<4s} "
            f"h={ev['h']:<4d} "
            f"rate={ev['rate']:.6g} "
            f"dt={ev['dt']:.6g}"
        )

# -----------------------------------------------------------
# MAIN LOOP (tubulin addition only)
# -----------------------------------------------------------

n_iterations    = 10000    # total Gillespie steps
                            # MATLAB: nIterations = 450000

snapshot_freq   = 250      # steps between output snapshots
                            # 1-2 seconds of real simulation time per snapshot
                            # MATLAB: cortime = 1000

print_freq    = 1000   # steps between console prints (0 = never print)
make_gif      = False    # rendering costs more wall time than the simulation;
                        # run_sim.py turns this off by default for sweeps
record_positions = True # record where every bound protein sits at each
                        # snapshot, plus a per-height lattice summary.
                        # Needed for kymographs and comet profiles.
max_sim_time  = None    # stop once this many SIMULATED seconds have elapsed.
                        # None = run the full n_iterations.
                        # A step buys less simulated time at high protein
                        # concentration (more events are protein events), so
                        # runs matched on step count are NOT matched on
                        # simulated time. Stop on this to compare fairly.


# -----------------------------------------------------------
# OBSERVABLES
# -----------------------------------------------------------

def oligomer_sizes() -> list:
    """
    Size in subunits of every connected cluster in protein_bonds, largest first.
    Monomers are not listed here: an unbonded protein has no entry in
    protein_bonds. Recover them as out_n_bound - sum(sizes).
    """
    seen, sizes = set(), []
    for node in protein_bonds:
        if node in seen:
            continue
        stack, size = [node], 0
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            size += 1
            stack.extend(protein_bonds.get(u, ()))
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def largest_oligomer() -> int:
    """Biggest cluster, 0 if nothing is bonded."""
    s = oligomer_sizes()
    return s[0] if s else 0


def count_bound_by_site_and_nuc() -> dict:
    """
    Live count of bound proteins keyed by (site_type, is_gdp), read straight off
    prot_sites rather than from the n_bound_prots_by_nuc counters, so the numbers
    written to disk do not depend on that bookkeeping being correct.
    """
    counts = {(s, gdp): 0
              for s in (SITE_LATTICE, SITE_EDGELAT2, SITE_EDGELONG2, SITE_EDGE3)
              for gdp in (0, 1)}
    for (g_idx, h) in bound_prots:
        site = int(st.prot_sites[g_idx, h, 0])
        nuc = int(st.prot_sites[g_idx, h, 1])
        key = (site, 0 if nuc == NUC_GTP else 1)
        if key in counts:
            counts[key] += 1
    return counts


def record_snapshot(step: int) -> None:
    """
    Append one row of observables. Previously this pushed a hardcoded 0 into
    out_n_bound and never touched the eight out_n_GTP_* / out_n_GDP_* lists,
    so a finished run carried no protein numbers at all.
    """
    out_step.append(int(step))
    out_time.append(float(st.time_elapsed))
    out_pf_lengths.append(pf_len.copy())
    out_n_bound.append(len(bound_prots))
    out_n_tethered.append(len(tethered_prots))
    out_n_bonds.append(sum(len(v) for v in protein_bonds.values()) // 2)
    sizes = oligomer_sizes()                      # one walk, used twice
    out_max_oligo.append(sizes[0] if sizes else 0)
    out_oligo_sizes.append(np.array(sizes, dtype=np.int16))

    if record_positions:
        # Per-height lattice summary. This is the same information the GIF
        # draws, stored as numbers instead of pixels: rendering is what costs
        # wall time, collecting is nearly free. Kymographs and comet profiles
        # can both be rebuilt from this later without re-running.
        H = int(pf_len.max())
        present = np.arange(H)[None, :] < pf_len[:, None]          # (n_pf, H)
        gdp = (st.MT_lattice[:, :H, 0] == 1) & present
        out_lat_ntub.append(present.sum(0).astype(np.int16))
        out_lat_ngdp.append(gdp.sum(0).astype(np.int16))

        if bound_prots:
            out_positions.append(np.array(
                [(g, h,
                  int(st.prot_sites[g, h, 0]),
                  int(st.prot_sites[g, h, 1]),
                  1 if (g, h) in protein_bonds else 0)
                 for (g, h) in bound_prots], dtype=np.int32))
        else:
            out_positions.append(np.zeros((0, 5), dtype=np.int32))

    c = count_bound_by_site_and_nuc()
    out_n_GTP_0.append(c[(SITE_LATTICE, 0)])
    out_n_GDP_0.append(c[(SITE_LATTICE, 1)])
    out_n_GTP_1.append(c[(SITE_EDGELAT2, 0)])
    out_n_GDP_1.append(c[(SITE_EDGELAT2, 1)])
    out_n_GTP_2.append(c[(SITE_EDGELONG2, 0)])
    out_n_GDP_2.append(c[(SITE_EDGELONG2, 1)])
    out_n_GTP_3.append(c[(SITE_EDGE3, 0)])
    out_n_GDP_3.append(c[(SITE_EDGE3, 1)])

def run_simulation():

    stop_reason = 'n_iterations'

    for step in range(1, n_iterations + 1):

        if max_sim_time is not None and st.time_elapsed >= max_sim_time:
            stop_reason = 'max_sim_time'
            print(f"Reached max_sim_time={max_sim_time} s at step {step - 1} "
                  f"(t={st.time_elapsed:.3f} s)")
            break

        # --- build candidate list ---
        candidates = (generate_tub_add_events()
                    + generate_tub_removal_events()
                    + generate_lat_bond_formation_events()
                    + generate_lat_bond_break_events()
                    + generate_prot_bind_events()
                    + generate_prot_remove_events()
                    + generate_prot_reattach_events()
                    + generate_prot_bond_form_events()
                    + generate_prot_bond_break_events())  

        # --- pick fastest event ---
        event = choose_next_event(candidates)
        if event is None:
            print(f"No candidates at step {step}, stopping.")
            stop_reason = 'no_candidates'
            break
        
        if print_freq > 0 and step % print_freq == 0:
            print(f"\n=== STEP {step} ===")  
            print_candidate_events(candidates)
            print_selected_event(event)

        # --- advance time ---
        st.time_elapsed += event['dt']

        # --- execute ---
        if event['event_type'] == 'tub_add':
            execute_tub_add(event)
        elif event['event_type'] == 'tub_removal':
            execute_tub_remove(event)
        elif event['event_type'] == 'lat_bond_form':
            execute_lat_bond_form(event)
        elif event['event_type'] == 'lat_bond_break':
            execute_lat_bond_break(event)
        elif event['event_type'].startswith('prot_bind'):
            execute_prot_bind(event)
        elif event['event_type'].startswith('prot_remove'):
            execute_prot_remove(event)
        elif event['event_type'].startswith('prot_reattach'):
            execute_prot_reattach(event)
        elif event['event_type'] == 'prot_bond_form':
            execute_prot_bond_form(event)
        elif event['event_type'] == 'prot_bond_break':
            execute_prot_bond_break(event)
            
        # A tethered subunit is only held by its bonds. Those break in several
        # places -- a bond-break event, or unbind_protein clearing a partner's
        # bonds on its way out -- so sweep once per step rather than trying to
        # catch every path.
        if tethered_prots:
            drop_unheld_tethers()

        # --- hydrolysis ---
        execute_hydrolysis(dt=event['dt'])

        # --- snapshot ---
        if step % snapshot_freq == 0:
            record_snapshot(step)

            if print_freq > 0:
                print(f"  After execution:")
                print(f"    t={st.time_elapsed:.4f}s")
                print(f"    pf_len       = {list(pf_len)}")
                print(f"    highest_lat  = {list(highest_lat)}")
                print(f"    mean_len={pf_len.mean():.1f}  max={pf_len.max()}  min={pf_len.min()}")

            if make_gif:
                plot_pf_lengths_and_lattice_occupancy()

    # final row, so the end state is always recorded even if the loop stopped early
    if not out_step or out_step[-1] != step:
        record_snapshot(step)
    return stop_reason

def save_gif(path: str = None) -> str:
    """Write the collected frames. Returns the path, or None if there were none."""
    if not frames:
        return None
    if path is None:
        path = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=500,   # ms per frame
        loop=0
    )
    print(f"Saved {path} ({len(frames)} frames)")
    return path


def print_height_report() -> None:
    print(f"\nLattice height: {st.MT_lattice.shape[1]} rows "
          f"({st.n_height_grows} reallocation(s) from {array_len_init}); "
          f"tallest protofilament = {int(pf_len.max())}")


def print_final_state() -> None:
    print("\n=== MT_lattice final state ===")
    for pf in range(n_pf):
        print(f"\n  PF {pf} (len={pf_len[pf]}):")
        print(f"  {'h':>4}  {'hydro':>6}  {'r_bonds':>8}")
        for h in range(pf_len[pf]):
            hydro  = int(st.MT_lattice[pf, h, 0])
            rbonds = int(st.MT_lattice[pf, h, 1])
            print(f"  {h:>4}  {hydro:>6}  {rbonds:>8}")


if __name__ == "__main__":
    import sys
    import time as _time
    from save_outputs import save_all

    # Location where the run writes its data
    outdir = sys.argv[1] if len(sys.argv) > 1 else \
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    _t0 = _time.time()
    stop_reason = run_simulation()
    _wall = _time.time() - _t0

    save_all(outdir, sys.modules[__name__], stop_reason=stop_reason,
             wall_seconds=_wall)

    if make_gif:
        save_gif(os.path.join(outdir, "simulation.gif"))
    print_height_report()
