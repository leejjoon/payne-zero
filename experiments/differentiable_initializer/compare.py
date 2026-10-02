"""Atmosphere-level test: emulator AD Jacobian vs finite differences of re-converged atmospheres.

Metrics follow DSS Gate G1 / V1.3 (rel-L2 < 5% and cosine > 0.995 to pass), over the
line-forming band -3 <= log tau_Ross <= 1.  Two references per cell:

  * richardson: (4 J(h) - J(2h)) / 3, scored only on layers where |J(h) - J(2h)| / |J| < 5%
    (the DSS V1.3 construction; uses the +-2h solves, so an unconverged 2h point degrades it);
  * central_h : (s(+h) - s(-h)) / 2h on every band layer, using only the +-h solves, reported
    with the reference's own uncertainty, estimated as the disagreement of the two one-sided
    differences (s(+h) - s0)/h and (s0 - s(-h))/h.  An emulator error comparable to that spread
    is inside the reference's noise and cannot be called a failure.

    python compare.py results/runs/sun results/runs/metalpoor_giant
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

import fd_reference as F
import pz_jax_init as J

FIELDS = ("T", "P_gas", "m", "n_e")
BAND = (-3.0, 1.0)
STABILITY = 0.05
SUB_BANDS = ((-5.0, -3.0), (-3.0, -1.0), (-1.0, 1.0), (1.0, 2.0))


def load(d: Path, name, warn=True):
    z = np.load(d / f"{name}.npz")
    if warn and not bool(z["converged"]):
        dt = json.loads(str(z["diagnostics"])).get("deep_layer_relative_temperature_change")
        print(f"  note: {d.name}/{name} stopped at the iteration cap (final deep dT/T = {dt:.1e})")
    return np.log(np.stack([z[f] for f in FIELDS]))


def rel_cos(a, r):
    return (float(np.linalg.norm(a - r) / np.linalg.norm(r)),
            float(a @ r / (np.linalg.norm(a) * np.linalg.norm(r))))


def analyze(d: Path, ji):
    s0 = load(d, "base")
    labels = np.array(F.BASES[d.name])
    J_emu = np.asarray(ji.log_state_jacobian(labels))
    lt = F.LOG_TAU
    band = (lt >= BAND[0]) & (lt <= BAND[1])
    report = {}
    for i, lab in enumerate(F.LABELS):
        h = F.STEPS[lab]
        sp1, sm1 = load(d, f"{lab}_p1"), load(d, f"{lab}_m1")
        sp2, sm2 = load(d, f"{lab}_p2"), load(d, f"{lab}_m2")
        Jh = (sp1 - sm1) / (2 * h)
        J2h = (sp2 - sm2) / (4 * h)
        Jrich = (4 * Jh - J2h) / 3
        stab = np.abs(Jh - J2h) / np.maximum(np.abs(Jrich), 1e-30)
        Jplus, Jminus = (sp1 - s0) / h, (s0 - sm1) / h
        for f, field in enumerate(FIELDS):
            cell = {}
            ok = band & (stab[f] < STABILITY)
            if ok.sum() >= 5:
                rel, cos = rel_cos(J_emu[f, ok, i], Jrich[f, ok])
                cell["richardson"] = {"rel_l2": rel, "cosine": cos, "n_trusted_layers": int(ok.sum()),
                                      "median_ratio": float(np.median(J_emu[f, ok, i] / Jrich[f, ok])),
                                      "fd_h_vs_2h_median": float(np.median(stab[f, ok])),
                                      "passes_G1": rel < 0.05 and cos > 0.995}
            else:
                cell["richardson"] = {"n_trusted_layers": int(ok.sum())}
            rel, cos = rel_cos(J_emu[f, band, i], Jh[f, band])
            spread = float(np.linalg.norm(Jplus[f, band] - Jminus[f, band]) / np.linalg.norm(Jh[f, band]))
            by_depth = {}
            for lo, hi in SUB_BANDS:
                sb = (lt >= lo) & (lt <= hi)
                by_depth[f"[{lo:g},{hi:g}]"] = rel_cos(J_emu[f, sb, i], Jh[f, sb])[0]
            cell["central_h"] = {"rel_l2": rel, "cosine": cos, "one_sided_spread": spread,
                                 "rel_l2_by_log_tau": by_depth}
            report[f"{field}/{lab}"] = cell
    v = np.asarray(ji.dss_state(labels))
    vals = {field: float(np.median(np.abs(np.log(v[f, band]) - s0[f, band]))) for f, field in enumerate(FIELDS)}
    return report, vals


def main(dirs):
    ji = J.JaxInitializer()
    full = {"band_log_tau": BAND, "stability_threshold": STABILITY, "bases": {}}
    for d in map(Path, dirs):
        rep, vals = analyze(d, ji)
        full["bases"][d.name] = {"labels": F.BASES[d.name], "value_median_abs_dln": vals, "jacobian": rep}
        print(f"\n== {d.name}  emulator value error, median |d ln|: " + "  ".join(f"{k} {v:.1e}" for k, v in vals.items()))
        print(f"{'cell':12s} | {'Richardson':^26s} | {'central +-h':^30s}")
        print(f"{'':12s} | {'rel':>6s} {'cos':>7s} {'n':>3s} {'pass':>5s} | {'rel':>6s} {'cos':>7s} {'ref-spread':>11s}")
        for k, c in rep.items():
            r, h = c["richardson"], c["central_h"]
            rr = (f"{r['rel_l2']:6.3f} {r['cosine']:7.4f} {r['n_trusted_layers']:3d} {'PASS' if r['passes_G1'] else 'fail':>5s}"
                  if "rel_l2" in r else f"{'(' + str(r['n_trusted_layers']) + ' layers)':>26s}")
            print(f"{k:12s} | {rr} | {h['rel_l2']:6.3f} {h['cosine']:7.4f} {h['one_sided_spread']:11.3f}")
    out = J.RESULTS / "atmosphere_jacobian.json"
    out.write_text(json.dumps(full, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
