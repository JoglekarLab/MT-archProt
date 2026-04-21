# =============================================================
# simulation.py
# =============================================================
# Main Gillespie simulation loop for the microtubule + EB1-like model.
# This file imports the initialized state from initialization.py.

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
from PIL import Image

from params import *
from helpers import classify_pocket
from initialization import *
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
    Walk up from highest_full_GDP checking if all PFs are GDP at each row.
    Stop at the first row where any PF is still GTP.
    up_to: the height of the dimer that just hydrolyzed.
    """
    global highest_full_GDP
    for hh in range(highest_full_GDP + 1, hydrolyzed_hh + 1):
        if all(MT_lattice[pf, hh, 0] == 1 for pf in range(n_pf)):
            highest_full_GDP = hh
        else:
            break

def execute_hydrolysis(dt: float) -> None:
    """
    Only buried dimers hydrolyze, so the loop to sample a waiting time goes backwards from the second to last dimer of each PF.
    If that waiting time < dt, the dimer hydrolyzes.
    Hydrolysis check called once per iteration after the winning event fires. This is computationally more efficient than putting all hydrolysis events in the candidate list for Guillespie.
    """
    for pf in range(n_pf):
        for hh in range(pf_len[pf] - 2, highest_full_GDP, -1):

            if MT_lattice[pf, hh, 0] == 1:  # already GDP
                continue

            if sample_exponential_dt(k_hydrolysis) < dt:
                MT_lattice[pf, hh, 0] = 1   # GTP → GDP
                update_highest_full_GDP(hydrolyzed_hh=hh)
                refresh_local_environment_after_tubulin_change(pf, hh)
                
# -----------------------------------------------------------
# EVENT GENERATION
# -----------------------------------------------------------

def generate_tub_add_events():
    candidate_events = []
    for pf_idx in range(n_pf):
        hh = pf_len[pf_idx]  # next available height where a new tubulin can go
        if not height_in_bounds(hh):
            continue
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
    
    def removable_prots(site, nuc):
        return n_bound_prots_by_nuc[site][nuc] - n_bonded_prots_by_nuc[site][nuc]

    n_gtp_edge = (removable_prots(SITE_EDGELAT2,  NUC_GTP) +
                  removable_prots(SITE_EDGELONG2, NUC_GTP) +
                  removable_prots(SITE_EDGE3,     NUC_GTP))

    n_gdp_edge = (removable_prots(SITE_EDGELAT2,  NUC_MIXED) + removable_prots(SITE_EDGELAT2,  NUC_GDP) +
                  removable_prots(SITE_EDGELONG2, NUC_MIXED) + removable_prots(SITE_EDGELONG2, NUC_GDP) +
                  removable_prots(SITE_EDGE3,     NUC_MIXED) + removable_prots(SITE_EDGE3,     NUC_GDP))

    n_gtp_lattice = removable_prots(SITE_LATTICE, NUC_GTP)
    
    n_gdp_lattice = (removable_prots(SITE_LATTICE, NUC_MIXED) + removable_prots(SITE_LATTICE, NUC_GDP))

    if n_gtp_edge > 0:
        candidate_events.append(make_event('prot_remove_gtp_edge',    rate=koff_prot_GTP_1 * n_gtp_edge,    h=-1)) # Note h=-1 is just a placeholder, means nothing, it's just passed as the function needs it, but in reality the rate is for all sites
    if n_gdp_edge > 0:
        candidate_events.append(make_event('prot_remove_gdp_edge',    rate=koff_prot_GDP_1 * n_gdp_edge,    h=-1))
    if n_gtp_lattice > 0:
        candidate_events.append(make_event('prot_remove_gtp_lattice', rate=koff_prot_GTP_0 * n_gtp_lattice, h=-1))
    if n_gdp_lattice > 0:
        candidate_events.append(make_event('prot_remove_gdp_lattice', rate=koff_prot_GDP_0 * n_gdp_lattice, h=-1))

    return candidate_events

def generate_prot_bond_form_events():
    """
    One event per unbonded pair of neighboring bound proteins.
    g < g2 filter avoids generating each pair twice.
    """
    candidate_events = []
    for (g, h) in bound_prots:
        for (g2, h2) in get_protein_neighbors(g, h):
            if g2 <= g: # g2 <= g filter avoids generating each pair twice.
                continue
            if not get_pocket_is_bound(g2, h2):
                continue
            if (g2, h2) in protein_bonds.get((g, h), set()): # Already bonded with each other
                continue
            event = make_event('prot_bond_form', rate=k_prot_bond_form, h=h, g=g)
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

    pf_len[pf] += 1 # extend the protofilament

    # new dimer is GTP by default (MT_lattice already zero-initialized)
    # MT_lattice[pf, h, 0] = 0  # explicit but redundant for fresh positions

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
    MT_lattice[pf, h, 0] = 0   # hydrolysis layer reset
    MT_lattice[pf, h, 1] = 0   # right-bond count reset

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
        MT_lattice[pf, h, 1] += 1
        if MT_lattice[pf, h, 1] == 2:   # seam needs both bonds to be "fully bonded"
            highest_lat[pf] = h
    else:
        if right_bond_count(pf, h) >= 1:
            raise RuntimeError(f"lat_bond_form: already bonded at PF {pf}, h={h}")
        MT_lattice[pf, h, 1] = 1
        highest_lat[pf] = h              # h is now the new highest bonded

    refresh_local_environment_after_tubulin_change(pf, h)


def execute_lat_bond_break(event: dict) -> None:
    pf = event['pf']
    h  = event['h']

    if right_bond_count(pf, h) == 0:
        raise RuntimeError(f"lat_bond_break: no right bond at PF {pf}, h={h}")

    MT_lattice[pf, h, 1] -= 1

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
        if prot_bond_count(g, h) > 0:   # NEW: bonded proteins cannot be removed
            continue
        site = int(prot_sites[g, h, 0])
        nuc  = int(prot_sites[g, h, 1])
        is_lattice = (site == SITE_LATTICE)
        is_gdp     = (nuc != NUC_GTP)
        if is_lattice == lattice_event and is_gdp == gdp_event:
            proteins.append((g, h))

    if not proteins:
        raise RuntimeError(f"execute_prot_remove: pool '{event['event_type']}' is empty at execution — counter mismatch")

    g, h = proteins[np.random.randint(len(proteins))]
    unbind_protein(g, h, event['event_type'])
    
    
def execute_prot_bond_form(event: dict) -> None:
    g1, h1, g2, h2 = event['g'], event['h'], event['g2'], event['h2']
    if not get_pocket_is_bound(g1, h1) or not get_pocket_is_bound(g2, h2):
        raise RuntimeError(f"execute_prot_bond_form: one of the pockets is not bound at execution — counter mismatch: (g1,h1)=({g1},{h1}), (g2,h2)=({g2},{h2})")
    if (g2, h2) in protein_bonds.get((g1, h1), set()):
        raise RuntimeError(f"execute_prot_bond_form: pockets already bonded at execution — counter mismatch: (g1,h1)=({g1},{h1}), (g2,h2)=({g2},{h2})")
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # --- left: PF lengths bar chart (no gaps) ---
    ax1.bar(np.arange(n_pf), pf_len.copy(), width=1.0, edgecolor='blue', linewidth=0.3)
    ax1.set_xlabel("Protofilament")
    ax1.set_ylabel("Length")
    ax1.set_title(f"PF lengths at t = {time_elapsed:.4f} s")
    
    # --- right: lattice occupancy with protein overlay and GTP/GDP coloring ---
    max_h = int(pf_len.max())
    lattice_img = np.zeros((max_h, n_pf), dtype=int)
    for pf in range(n_pf):
        for h in range(pf_len[pf]):
            if MT_lattice[pf, h, 0] == 0:
                lattice_img[h, pf] = 1   # GTP
            else:
                lattice_img[h, pf] = 2   # GDP

    cmap = matplotlib.colors.ListedColormap(['white', '#0a2472', '#4a90d9'])  # 0 = background (white), 1 = GTP (dark blue), 2 = GDP (medium blue)
    im = ax2.imshow(lattice_img, origin='lower', aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax2.set_xlabel("Protofilament")
    ax2.set_ylabel("Height")
    ax2.set_title(f"Lattice + protein occupancy at t = {time_elapsed:.4f} s")
    
    # fig.colorbar(im, ax=ax2, label="Tubulin present")
    # cbar = fig.colorbar(im, ax=ax2, ticks=[1, 2])
    # cbar.ax.set_yticklabels(['GTP', 'GDP'])
    
    # legend_elements = [
    #     Patch(facecolor='#4a90d9', label='GTP'),
    #     Patch(facecolor='#0a2472', label='GDP'),
    # ]
    legend_elements = [
        Patch(facecolor='#0a2472', label='GTP'),
        Patch(facecolor='#4a90d9', label='GDP'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    # overlay bound proteins as dots
    for g in range(1, n_pf):  # skip seam grooves 0 and n_pf
        for h in range(max_h):
            if prot_sites[g, h, 2] == 1:
                ax2.plot(g - 0.5, h, '.', color='#00FF00', markersize=8)  # g-0.5 centers the dot between the two PFs

    # overlay lateral bonds as horizontal lines between PF columns
    # each cell in imshow is centered at x=pf, so the gap between pf and pf+1 spans x=[pf+0.5, pf+1-0.5]
    # we draw a short line from pf+0.1 to pf+0.9 at height h
    for pf in range(n_pf - 1):  # PF0-11: right bond connects pf to pf+1
        for h in range(seed_length, pf_len[pf]):
            if right_bond_count(pf, h) > 0:
                ax2.plot([pf + 0.3, pf + 0.7], [h, h], color='orange', linewidth=1.5)

    # seam: PF12 bonds to PF0, drawn on the right edge of PF12 (x = n_pf - 1 + 0.5 area)
    # since PF0 is at x=0 and PF12 is at x=12, the seam wraps; mark it on PF12's right side
    for h in range(seed_length, pf_len[n_pf - 1]):
        bc = right_bond_count(n_pf - 1, h)
        if bc == 1:
            ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='yellow', linewidth=1.5)
        elif bc == 2:
            ax2.plot([n_pf - 1 + 0.3, n_pf - 1 + 0.45], [h, h], color='red', linewidth=1.5)

    # render to in-memory buffer and store as PIL image
    buf = io.BytesIO() # Creates a fake file that lives in RAM instead of on disk
    fig.savefig(buf, format='png', bbox_inches='tight') # Write the figure to the RAM buffer
    buf.seek(0)
    frames.append(Image.open(buf).copy()) # will read image from first line thanks to buf.seek(0)
    buf.close() # Frees the RAM used by the buffer
    plt.close(fig)
    
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

n_iterations = 1000
snapshot_freq = 10

def run_simulation():
    global time_elapsed

    for step in range(1, n_iterations + 1):

        print(f"\n=== STEP {step} ===")

        # --- build candidate list ---
        candidates = (generate_tub_add_events()
                    + generate_tub_removal_events()
                    + generate_lat_bond_formation_events()
                    + generate_lat_bond_break_events()
                    + generate_prot_bind_events()
                    + generate_prot_remove_events()
                    + generate_prot_bond_form_events()
                    + generate_prot_bond_break_events())    
        
        print_candidate_events(candidates)

        # --- pick fastest event ---
        event = choose_next_event(candidates)
        if event is None:
            print(f"No candidates at step {step}, stopping.")
            break

        print_selected_event(event)

        # --- advance time ---
        time_elapsed += event['dt']

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
        elif event['event_type'] == 'prot_bond_form':
            execute_prot_bond_form(event)
        elif event['event_type'] == 'prot_bond_break':
            execute_prot_bond_break(event)
            
        # --- hydrolysis ---
        execute_hydrolysis(dt=event['dt'])

        # --- snapshot ---
        if step % snapshot_freq == 0:
            out_time.append(time_elapsed)
            out_pf_lengths.append(pf_len.copy())
            out_n_bound.append(0)

            print(f"  After execution:")
            print(f"    t={time_elapsed:.4f}s")
            print(f"    pf_len       = {list(pf_len)}")
            print(f"    highest_lat  = {list(highest_lat)}")
            print(f"    mean_len={pf_len.mean():.1f}  max={pf_len.max()}  min={pf_len.min()}")

            plot_pf_lengths_and_lattice_occupancy()

run_simulation()

if frames:
    frames[0].save(
        "simulation.gif",
        save_all=True,
        append_images=frames[1:],
        duration=500,   # ms per frame
        loop=0
    )
    print(f"Saved simulation.gif ({len(frames)} frames)")
    
print("\n=== MT_lattice final state ===")
for pf in range(n_pf):
    print(f"\n  PF {pf} (len={pf_len[pf]}):")
    print(f"  {'h':>4}  {'hydro':>6}  {'r_bonds':>8}")
    for h in range(pf_len[pf]):
        hydro  = int(MT_lattice[pf, h, 0])
        rbonds = int(MT_lattice[pf, h, 1])
        print(f"  {h:>4}  {hydro:>6}  {rbonds:>8}")