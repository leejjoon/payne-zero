"""Supporting checks quoted in REPORT.md, written to results/checks.json.

  1. tau grid: the converged solver's own Rosseland depth vs the canonical grid log tau = -6.875 + 0.125 j
     (which is also the initializer's grid and DSS's canonical grid);
  2. tolerance sensitivity: the Sun base solved at 2e-5 and at 5e-6, compared with the 25 K Teff signal;
  3. label conventions: Payne Zero's solar abundances and alpha set vs DSS's dss/couple/labels.py
     (skipped if the DSS checkout is not at $DSS_REPO or ../differentiable_stellar_spectroscopy).

    python checks.py
"""
from __future__ import annotations

import json, os
from pathlib import Path

import numpy as np

import fd_reference as F
from pz_jax_init import REPO, RESULTS

FIELDS = ("T", "P_gas", "m", "n_e")


def tau_grid():
    out = {}
    for base in ("sun", "metalpoor_giant"):
        z = np.load(RESULTS / "runs" / base / "base.npz")
        d = np.abs(z["native_log_tau"] - F.LOG_TAU)
        out[base] = {"median_abs_dlogtau": float(np.median(d)), "max_abs_dlogtau": float(d.max())}
    return out


def tolerance():
    d = RESULTS / "runs" / "sun"
    loose, tight = np.load(d / "base_tol2e-5.npz"), np.load(d / "base.npz")
    band = (F.LOG_TAU >= -3) & (F.LOG_TAU <= 1)
    signal = np.abs(np.log(np.load(d / "teff_p1.npz")["T"]) - np.log(tight["T"]))[band]
    shift = np.abs(np.log(loose["T"]) - np.log(tight["T"]))[band]
    return {"median_abs_dlnT_2e-5_vs_5e-6": float(np.median(shift)),
            "median_abs_dlnT_for_25K_teff_step": float(np.median(signal)),
            "ratio": float(np.median(shift) / np.median(signal))}


def conventions():
    dss = Path(os.environ.get("DSS_REPO", REPO.parent / "differentiable_stellar_spectroscopy"))
    src = dss / "dss" / "couple" / "labels.py"
    if not src.exists():
        return {"skipped": f"{src} not found"}
    from payne_zero_atmosphere.warm_start import ALPHA_ELEMENT_ATOMIC_NUMBERS, SOLAR_METAL_LOG_ABUNDANCES_3_TO_99
    ns = {}
    exec(src.read_text().split("def number_fractions")[0].replace("from __future__ import annotations", ""), ns)
    return {"max_abs_solar_log_abundance_difference_Z3_99": float(np.max(np.abs(
                np.asarray(SOLAR_METAL_LOG_ABUNDANCES_3_TO_99) - ns["SOLAR_LOG_ABUND"]))),
            "alpha_elements_identical": tuple(ALPHA_ELEMENT_ATOMIC_NUMBERS) == tuple(ns["ALPHA_Z"]),
            "helium_fraction_dss": ns["HE_FRACTION"]}


if __name__ == "__main__":
    out = {"tau_grid": tau_grid(), "tolerance": tolerance(), "conventions": conventions()}
    (RESULTS / "checks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
