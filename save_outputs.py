"""
Write everything a run holds in memory to disk.

Call save_all(outdir) after run_simulation(). Nothing here changes the
simulation -- it only serialises what record_snapshot and the event log already
collected, which is otherwise discarded when the process exits.

Files written into outdir:

    meta.json         parameters, run settings, stop reason, timing
    timeseries.csv    one row per snapshot: counts, tip, all 13 PF lengths
    prot_events.csv   one row per binding event -- the single-molecule record
    positions.csv     one row per protein per snapshot, with distance behind tip
    oligomers.csv     one row per oligomer per snapshot (long format)
    lattice.npz       height x time arrays: tubulin, GDP, protein. Kymographs.
    final_state.npz   MT_lattice, prot_sites, pf_len, highest_lat at the end
    bonds_final.csv   the oligomer graph as it stood at the end
"""

import csv
import json
import os
import time

import numpy as np

import initialization as st
from params import *


def _pad(list_of_1d):
    """Ragged per-snapshot rows -> one (n_snapshots, max_height) array."""
    if not list_of_1d:
        return np.zeros((0, 0), dtype=np.int16)
    H = max(int(a.shape[0]) for a in list_of_1d)
    out = np.zeros((len(list_of_1d), H), dtype=np.int16)
    for i, a in enumerate(list_of_1d):
        out[i, :a.shape[0]] = a
    return out


def _tip_of(pf_lengths):
    return int(np.max(pf_lengths))


def save_meta(outdir, label, stop_reason, wall_seconds, sim_module):
    path = os.path.join(outdir, 'meta.json')
    meta = {
        'label': label,
        'stop_reason': stop_reason,
        'wall_seconds': round(float(wall_seconds), 2),
        'sim_time_s': float(st.time_elapsed),
        'n_snapshots': len(st.out_step),
        'steps_run': int(st.out_step[-1]) if st.out_step else 0,
        'lattice_rows': int(st.MT_lattice.shape[1]),
        'height_grows': int(st.n_height_grows),
        'tip_final': int(st.pf_len.max()),
        'run_settings': {
            'n_iterations': sim_module.n_iterations,
            'snapshot_freq': sim_module.snapshot_freq,
            'max_sim_time': sim_module.max_sim_time,
            'record_positions': sim_module.record_positions,
            'make_gif': sim_module.make_gif,
        },
        'params': {
            'n_pf': n_pf, 'seed_length': seed_length,
            'conc_tubGTP': conc_tubGTP, 'conc_prot': conc_prot,
            'conc_prot_tethered_nM': conc_prot_tethered_nM,
            'kon_tub': kon_tub, 'koff_tubGTP': koff_tubGTP, 'koff_tubGDP': koff_tubGDP,
            'k_hydrolysis': k_hydrolysis,
            'k_lateralbond': k_lateralbond, 'k_lateralbondSeam': k_lateralbondSeam,
            'k_lateralbreak_TT': k_lateralbreak_TT,
            'k_lateralbreak_TD': k_lateralbreak_TD,
            'k_lateralbreak_DD': k_lateralbreak_DD,
            'lateral_stabilization_factor': lateral_stabilization_factor,
            'interaction_range': interaction_range,
            'k_prot_bond_form_same_h': k_prot_bond_form_same_h,
            'k_prot_bond_form_diff_h': k_prot_bond_form_diff_h,
            'k_prot_bond_break': k_prot_bond_break,
            'prot_bond_stabilization_factor': prot_bond_stabilization_factor,
            'kon_prot_0': kon_prot_0, 'kon_prot_1': kon_prot_1,
            'koff_prot_GTP_0': koff_prot_GTP_0, 'koff_prot_GDP_0': koff_prot_GDP_0,
            'koff_prot_GTP_1': koff_prot_GTP_1, 'koff_prot_GDP_1': koff_prot_GDP_1,
        },
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)
    return path


def save_timeseries(outdir):
    """One row per snapshot. Small file, enough for every time-course plot."""
    path = os.path.join(outdir, 'timeseries.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'time_s', 'tip', 'mean_pf_len',
                    'n_bound', 'n_tethered', 'n_bonds', 'max_oligo',
                    'n_GTP_lattice', 'n_GDP_lattice', 'n_GTP_edgelat2', 'n_GDP_edgelat2',
                    'n_GTP_edgelong2', 'n_GDP_edgelong2', 'n_GTP_edge3', 'n_GDP_edge3']
                   + [f'pf{i}' for i in range(n_pf)])
        for i in range(len(st.out_step)):
            pf = st.out_pf_lengths[i]
            w.writerow([st.out_step[i], f'{st.out_time[i]:.6f}',
                        _tip_of(pf), f'{float(np.mean(pf)):.3f}',
                        st.out_n_bound[i], st.out_n_tethered[i],
                        st.out_n_bonds[i], st.out_max_oligo[i],
                        st.out_n_GTP_0[i], st.out_n_GDP_0[i],
                        st.out_n_GTP_1[i], st.out_n_GDP_1[i],
                        st.out_n_GTP_2[i], st.out_n_GDP_2[i],
                        st.out_n_GTP_3[i], st.out_n_GDP_3[i]]
                       + [int(x) for x in pf])
    return path


def save_prot_events(outdir):
    """
    The single-molecule record: one row per protein that ever bound.

    This is the closest thing the model has to what a TIRF experiment measures --
    every dwell time, where it landed, how it left. 'h' is the height it bound
    at and 'h_now' where it ended up, so h_now - h is how far it migrated.
    't_off' empty means it was still attached when the run stopped, so those
    rows are right-censored and should be excluded from dwell-time fits.
    """
    path = os.path.join(outdir, 'prot_events.csv')
    cols = ['g_idx', 'h', 'h_now', 'n_hops', 'n_detach',
            'site_type_on', 'nuc_state_on', 'site_type_now', 'nuc_state_now',
            't_on', 't_off', 'dwell_s', 'removal']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in st.prot_events:
            t_off = e.get('t_off')
            dwell = '' if t_off is None else f"{t_off - e['t_on']:.6f}"
            w.writerow([e['g_idx'], e['h'], e.get('h_now', e['h']),
                        e.get('n_hops', 0), e.get('n_detach', 0),
                        e['site_type_on'], e['nuc_state_on'],
                        e['site_type_now'], e['nuc_state_now'],
                        f"{e['t_on']:.6f}", '' if t_off is None else f"{t_off:.6f}",
                        dwell, e.get('removal') or ''])
    return path


def save_positions(outdir):
    """
    Every protein on the lattice at every snapshot, with how far behind the tip
    it sits. dist_from_local_tip uses the length of the protofilament beside the
    groove, not the longest one, so a tapered tip does not smear the comet.
    """
    if not st.out_positions:
        return None
    path = os.path.join(outdir, 'positions.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'time_s', 'groove', 'height', 'site_type',
                    'nuc_state', 'is_bonded', 'dist_from_local_tip', 'dist_from_max_tip'])
        for i, arr in enumerate(st.out_positions):
            pf = st.out_pf_lengths[i]
            step, t = st.out_step[i], st.out_time[i]
            tip_max = _tip_of(pf)
            for g, h, site, nuc, bonded in arr:
                # Groove g sits between PF g-1 and PF g, and its local tip is the
                # TALLER of the two. Not the shorter: a SITE_EDGELONG2 pocket is
                # two tubulins stacked on one protofilament with nothing opposite,
                # so the short side can be far below the protein and min() puts
                # 20% of rows above their own tip.
                local = max(int(pf[g - 1]), int(pf[g])) if 1 <= g < n_pf else tip_max
                w.writerow([step, f'{t:.6f}', int(g), int(h), int(site), int(nuc),
                            int(bonded), local - 1 - int(h), tip_max - 1 - int(h)])
    return path


def save_oligomers(outdir):
    """
    One row per oligomer per snapshot, long format.

    Monomers are NOT rows here -- an unbonded protein has no entry in
    protein_bonds, so it is not a cluster. Recover them per snapshot as
    n_bound - (sum of size over that step's rows); n_bound is in timeseries.csv.
    """
    if not st.out_oligo_sizes:
        return None
    path = os.path.join(outdir, 'oligomers.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'time_s', 'size'])
        for i, sizes in enumerate(st.out_oligo_sizes):
            step, t = st.out_step[i], st.out_time[i]
            for s in sizes:
                w.writerow([step, f'{t:.6f}', int(s)])
    return path


def save_lattice(outdir):
    """
    Height x time arrays -- everything the GIF draws, as numbers.

    n_tubulin  how many of the 13 PFs reach each height
    n_gdp      how many of those are GDP
    n_protein  bound proteins at each height
    Render kymographs from these instead of watching an animation, and note the
    time axis is NOT linear in step: use time_s for the axis or the picture lies.
    """
    if not st.out_lat_ntub:
        return None
    n_tub = _pad(st.out_lat_ntub)
    n_gdp = _pad(st.out_lat_ngdp)

    H = n_tub.shape[1]
    n_prot = np.zeros((len(st.out_positions), H), dtype=np.int16)
    for i, arr in enumerate(st.out_positions):
        for _, h, _, _, _ in arr:
            if 0 <= h < H:
                n_prot[i, h] += 1

    path = os.path.join(outdir, 'lattice.npz')
    np.savez_compressed(
        path,
        step=np.array(st.out_step, dtype=np.int64),
        time_s=np.array(st.out_time, dtype=np.float64),
        pf_len=np.array(st.out_pf_lengths, dtype=np.int32),
        tip=np.array([_tip_of(p) for p in st.out_pf_lengths], dtype=np.int32),
        n_tubulin=n_tub,
        n_gdp=n_gdp,
        n_protein=n_prot,
    )
    return path


def save_final_state(outdir):
    """The lattice itself at the end, plus the oligomer graph."""
    npz = os.path.join(outdir, 'final_state.npz')
    np.savez_compressed(npz,
                        MT_lattice=st.MT_lattice,
                        prot_sites=st.prot_sites,
                        pf_len=st.pf_len,
                        highest_lat=st.highest_lat)

    csv_path = os.path.join(outdir, 'bonds_final.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['groove_1', 'height_1', 'groove_2', 'height_2'])
        seen = set()
        for (g1, h1), partners in st.protein_bonds.items():
            for (g2, h2) in partners:
                key = tuple(sorted([(g1, h1), (g2, h2)]))
                if key in seen:
                    continue
                seen.add(key)
                w.writerow([key[0][0], key[0][1], key[1][0], key[1][1]])
    return npz, csv_path


def save_all(outdir, sim_module, label=None, stop_reason=None, wall_seconds=0.0,
             quiet=False):
    """Write every output. Returns the list of paths written."""
    os.makedirs(outdir, exist_ok=True)
    label = label or os.path.basename(os.path.normpath(outdir))

    written = [
        save_meta(outdir, label, stop_reason, wall_seconds, sim_module),
        save_timeseries(outdir),
        save_prot_events(outdir),
        save_positions(outdir),
        save_oligomers(outdir),
        save_lattice(outdir),
    ]
    written.extend(save_final_state(outdir))
    written = [p for p in written if p]

    if not quiet:
        print(f"\nWrote {len(written)} files to {outdir}/")
        for p in written:
            print(f"  {os.path.basename(p):<18} {os.path.getsize(p)/1024:>9.1f} KB")
    return written
