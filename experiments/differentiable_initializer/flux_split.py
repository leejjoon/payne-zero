"""Split the [M/H] and [alpha/M] flux Jacobians into a direct and an atmosphere-mediated part.

A composition label changes the spectrum two ways: directly, through the abundances in the line
and continuum opacity, and indirectly, by changing the atmosphere.  flux_jacobian.py gives both arms
identical abundances, so the direct part is common to reference and emulator and flatters the
emulator's score.  Here the direct part is computed explicitly (base atmosphere held fixed, abundances
at l +- h) and removed from both arms, and the emulator is scored on the atmosphere part alone.

    python flux_split.py results/runs/sun        # needs flux_jac_{mh,am}.npz from flux_jacobian.py
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

import fd_reference as F
import flux_jacobian as FJ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_dir")
    p.add_argument("--wl", type=float, nargs=2, default=(1550.0, 1600.0))
    p.add_argument("--r-grid", type=float, default=100000.0)
    a = p.parse_args()
    d = Path(a.base_dir)
    xi = F.BASES[d.name][4]
    base = FJ.native(np.load(d / "base.npz"))
    out = {}
    for lab in ("mh", "am"):
        z = np.load(d / f"flux_jac_{lab}.npz")
        h = F.STEPS[lab]
        ab_p = FJ.abundances(np.load(d / f"{lab}_p1.npz")["labels"])
        ab_m = FJ.abundances(np.load(d / f"{lab}_m1.npz")["labels"])
        _, fp = FJ.flux(base, ab_p, xi, a)
        _, fm = FJ.flux(base, ab_m, xi, a)
        direct = (fp - fm) / (2 * h)
        r, e = z["ref"] - direct, z["emu"] - direct
        out[lab] = {
            "atmosphere_share_of_total": float(np.linalg.norm(r) / np.linalg.norm(z["ref"])),
            "rel_l2_emulator_on_atmosphere_part": float(np.linalg.norm(e - r) / np.linalg.norm(r)),
            "cosine_emulator_on_atmosphere_part": float(e @ r / (np.linalg.norm(e) * np.linalg.norm(r))),
        }
        print(f"{d.name}/{lab}: " + "  ".join(f"{k} {v:.4f}" for k, v in out[lab].items()), flush=True)
    (d / "flux_split.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
