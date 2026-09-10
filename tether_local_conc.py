"""
Local concentration (uM) seen by a tethered protein subunit, for choosing
conc_prot_tethered_nM in params.py. Nothing imports this file.

A bonded subunit that lets go of its pocket is still held by the ring on its
linker, so it rebinds against this concentration rather than bulk conc_prot.
Values are the exact freely-jointed chain, not a Gaussian, which overestimates
by ~2x at these extensions. Good to a factor of two.

Assumed: MT 25 nm, CH domain 3 nm on the surface, dimer rise 8 nm, ring midway
between two rows, 0.35 nm/residue contour, 0.90 nm Kuhn length.


# --- 28 nm ring ---   radial gap 0.00 nm, reach 4.000 nm
#     aa   contour   r/Lc      C_local
#     10    3.50 nm   1.14   unreachable
#     11    3.85 nm   1.04   unreachable
#     12    4.20 nm   0.95      14.3 uM
#     13    4.55 nm   0.88      84.5 uM
#     14    4.90 nm   0.82     172.4 uM
#     15    5.25 nm   0.76     267.9 uM
#     16    5.60 nm   0.71     366.3 uM
#     17    5.95 nm   0.67     468.0 uM   <- default
#     18    6.30 nm   0.63     576.9 uM
#     19    6.65 nm   0.60     680.3 uM
#     20    7.00 nm   0.57     784.4 uM
#     21    7.35 nm   0.54     884.6 uM
#     22    7.70 nm   0.52     978.6 uM
#     23    8.05 nm   0.50    1071.4 uM
#     24    8.40 nm   0.48    1155.0 uM
#     25    8.75 nm   0.46    1235.1 uM

# --- 31 nm ring ---   radial gap 1.50 nm, reach 4.272 nm
#     aa   contour   r/Lc      C_local
#     10    3.50 nm   1.22   unreachable
#     11    3.85 nm   1.11   unreachable
#     12    4.20 nm   1.02   unreachable
#     13    4.55 nm   0.94       9.8 uM
#     14    4.90 nm   0.87      46.7 uM
#     15    5.25 nm   0.81      97.9 uM
#     16    5.60 nm   0.76     156.2 uM
#     17    5.95 nm   0.72     219.2 uM   <- default
#     18    6.30 nm   0.68     288.8 uM
#     19    6.65 nm   0.64     359.9 uM
#     20    7.00 nm   0.61     434.5 uM
#     21    7.35 nm   0.58     509.5 uM
#     22    7.70 nm   0.55     582.5 uM
#     23    8.05 nm   0.53     656.2 uM
#     24    8.40 nm   0.51     725.4 uM
#     25    8.75 nm   0.49     793.2 uM

# --- 33 nm ring ---   radial gap 2.50 nm, reach 4.717 nm
#     aa   contour   r/Lc      C_local
#     10    3.50 nm   1.35   unreachable
#     11    3.85 nm   1.23   unreachable
#     12    4.20 nm   1.12   unreachable
#     13    4.55 nm   1.04   unreachable
#     14    4.90 nm   0.96       0.6 uM
#     15    5.25 nm   0.90       8.7 uM
#     16    5.60 nm   0.84      25.2 uM
#     17    5.95 nm   0.79      47.9 uM   <- default
#     18    6.30 nm   0.75      75.8 uM
#     19    6.65 nm   0.71     107.1 uM
#     20    7.00 nm   0.67     142.7 uM
#     21    7.35 nm   0.64     181.5 uM
#     22    7.70 nm   0.61     222.2 uM
#     23    8.05 nm   0.59     265.3 uM
#     24    8.40 nm   0.56     308.3 uM
#     25    8.75 nm   0.54     352.3 uM

Regenerate for other geometry:  python tether_local_conc.py --ring 30 --aa 14 20
"""

import argparse

import numpy as np

AVOGADRO = 6.02214076e23
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.0 renamed trapz

MT_DIAMETER_NM = 25.0
CH_DOMAIN_NM   = 3.0
DIMER_RISE_NM  = 8.0
AA_CONTOUR_NM  = 0.35
KUHN_NM        = 0.90
DEFAULT_N_AA   = 17

TABLES_uM = {
    28.0: {12: 14.3, 13: 84.5, 14: 172.4, 15: 267.9, 16: 366.3, 17: 468.0,
           18: 576.9, 19: 680.3, 20: 784.4, 21: 884.6, 22: 978.6, 23: 1071.4,
           24: 1155.0, 25: 1235.1},
    31.0: {13: 9.8, 14: 46.7, 15: 97.9, 16: 156.2, 17: 219.2, 18: 288.8,
           19: 359.9, 20: 434.5, 21: 509.5, 22: 582.5, 23: 656.2, 24: 725.4,
           25: 793.2},
    33.0: {14: 0.6, 15: 8.7, 16: 25.2, 17: 47.9, 18: 75.8, 19: 107.1,
           20: 142.7, 21: 181.5, 22: 222.2, 23: 265.3, 24: 308.3, 25: 352.3},
}


def tether_reach_nm(ring_diameter_nm, mt_diameter_nm=MT_DIAMETER_NM,
                    ch_domain_nm=CH_DOMAIN_NM, dimer_rise_nm=DIMER_RISE_NM):
    """Ring anchor to either of the two rows it straddles."""
    gap = abs(ring_diameter_nm / 2.0 - (mt_diameter_nm / 2.0 + ch_domain_nm / 2.0))
    return float(np.hypot(gap, dimer_rise_nm / 2.0))


def freely_jointed_density(r_nm, n_seg, b_nm, k_max=400.0, n_k=200_000):
    """
    End-to-end density (nm^-3) for n_seg random-direction segments of length
    b_nm:  P(r) = 1/(2 pi^2 r) INT k sin(kr) [sin(kb)/(kb)]^n dk. Exact.
    """
    if r_nm <= 0.0 or r_nm >= n_seg * b_nm:
        return 0.0
    k = np.linspace(1e-9, k_max, n_k)
    phi = (np.sin(k * b_nm) / (k * b_nm)) ** n_seg
    return float(_trapz(k * np.sin(k * r_nm) * phi, k) / (2.0 * np.pi ** 2 * r_nm))


def local_conc_uM(reach_nm, n_aa, aa_contour_nm=AA_CONTOUR_NM, kuhn_nm=KUHN_NM):
    """
    Local concentration (uM), 0 if the chain cannot span reach_nm. Interpolates
    between whole segment counts, else the value jumps as the count rounds.
    """
    contour = n_aa * aa_contour_nm
    if reach_nm >= contour:
        return 0.0
    x = contour / kuhn_nm
    n_lo = max(2, int(np.floor(x)))
    w = min(max(x - n_lo, 0.0), 1.0)
    c_lo = freely_jointed_density(reach_nm, n_lo, contour / n_lo)
    c_hi = freely_jointed_density(reach_nm, n_lo + 1, contour / (n_lo + 1))
    if c_lo <= 0.0 and c_hi <= 0.0:
        return 0.0
    if c_lo <= 0.0:
        density = c_hi
    elif c_hi <= 0.0:
        density = c_lo
    else:
        density = float(np.exp((1.0 - w) * np.log(c_lo) + w * np.log(c_hi)))
    return density * 1e24 / AVOGADRO * 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--ring', type=float, nargs='+', default=[28.0, 31.0, 33.0])
    ap.add_argument('--aa', type=int, nargs=2, default=[10, 25], metavar=('LO', 'HI'))
    ap.add_argument('--mt', type=float, default=MT_DIAMETER_NM)
    ap.add_argument('--ch', type=float, default=CH_DOMAIN_NM)
    ap.add_argument('--rise', type=float, default=DIMER_RISE_NM)
    ap.add_argument('--contour', type=float, default=AA_CONTOUR_NM)
    ap.add_argument('--kuhn', type=float, default=KUHN_NM)
    a = ap.parse_args()

    for D in a.ring:
        reach = tether_reach_nm(D, a.mt, a.ch, a.rise)
        gap = abs(D / 2.0 - (a.mt / 2.0 + a.ch / 2.0))
        print(f"\n# --- {D:.0f} nm ring ---   radial gap {gap:.2f} nm, reach {reach:.3f} nm")
        print(f"#     {'aa':>2}   {'contour':>7}   {'r/Lc':>4}      {'C_local':>7}")
        for n_aa in range(a.aa[0], a.aa[1] + 1):
            contour = n_aa * a.contour
            if reach >= contour:
                print(f"#     {n_aa:>2}   {contour:>4.2f} nm   {reach/contour:>4.2f}   unreachable")
                continue
            c = local_conc_uM(reach, n_aa, a.contour, a.kuhn)
            mark = "   <- default" if n_aa == DEFAULT_N_AA else ""
            print(f"#     {n_aa:>2}   {contour:>4.2f} nm   {reach/contour:>4.2f}   {c:>7.1f} uM{mark}")


if __name__ == "__main__":
    main()
