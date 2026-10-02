"""Does the emulator's atmosphere Jacobian give the right *spectrum* Jacobian?

For each label, two central differences of normalized flux with identical abundances on both arms:
  reference: atmospheres re-converged at l +- h                                     (truth)
  emulator : converged base atmosphere moved along the emulator's AD tangent, +- h  (what a
             differentiable fast path would give, with the solver supplying the value)
A third arm moves only T along the emulator tangent and P_gas, m, n_e along the reference,
to attribute any error to temperature or to the pressure-like fields.

    python flux_jacobian.py results/runs/sun --wl 1550 1600 --r-grid 100000
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

import fd_reference as F
import pz_jax_init as J

FIELDS = ("T", "P_gas", "m", "n_e")


def abundances(labels):
    from payne_zero_atmosphere import linear_elemental_abundances
    from payne_zero_atmosphere.warm_start import emulator_warm_start_model
    t, g, mh, am, xi = labels
    atm, _ = emulator_warm_start_model(effective_temperature=t, log_surface_gravity=g, metallicity=mh,
                                       alpha_enhancement=am, microturbulence_km_s=xi, device="cpu")
    return linear_elemental_abundances(atm)


def flux(cols, abund, xi, a):
    from payne_zero_synthesis import build_structured_atmosphere, synthesize
    s = build_structured_atmosphere(temperature=cols["T"], column_mass=cols["m"], gas_pressure=cols["P_gas"],
                                    electron_density=cols["n_e"], elemental_abundances=abund,
                                    microturbulence=xi * 1e5, device="cpu", dtype="float64")
    sp = synthesize(s, wavelength_start_nm=a.wl[0], wavelength_end_nm=a.wl[1], resolution=a.r_grid,
                    device="cpu", dtype="float64")
    return np.asarray(sp.wavelength_nm), np.asarray(sp.normalized_flux)


def native(z):
    return {f: np.asarray(z[f + "_native"], float) for f in FIELDS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_dir")
    p.add_argument("--wl", type=float, nargs=2, default=(1550.0, 1600.0))
    p.add_argument("--r-grid", type=float, default=100000.0)
    p.add_argument("--labels", default="teff,logg,mh,am")
    a = p.parse_args()
    d = Path(a.base_dir)
    base_labels = np.array(F.BASES[d.name])
    xi = base_labels[4]
    Je = np.asarray(J.JaxInitializer().log_state_jacobian(base_labels))   # (4, 80, 5)
    b = native(np.load(d / "base.npz"))
    results = {}
    for lab in a.labels.split(","):
        i = F.LABELS.index(lab)
        h = F.STEPS[lab]
        zp, zm = np.load(d / f"{lab}_p1.npz"), np.load(d / f"{lab}_m1.npz")
        def usable(z):  # converged, or stopped within 4x of the 5e-6 target (still 25x tighter than production)
            dt = json.loads(str(z["diagnostics"])).get("deep_layer_relative_temperature_change", 1.0)
            return bool(z["converged"]) or float(dt) < 2e-5
        if not (usable(zp) and usable(zm)):
            print(f"skip {lab}: +-h not both usable"); continue
        rp, rm = native(zp), native(zm)
        ab_p, ab_m = abundances(zp["labels"]), abundances(zm["labels"])

        def along(sign, t_only=False):
            out = {}
            for f, field in enumerate(FIELDS):
                if t_only and field != "T":
                    out[field] = (rp if sign > 0 else rm)[field]
                else:
                    out[field] = b[field] * np.exp(sign * h * Je[f, :, i])
            return out

        wl, fp = flux(rp, ab_p, xi, a); _, fm = flux(rm, ab_m, xi, a)
        dref = (fp - fm) / (2 * h)
        _, ep = flux(along(+1), ab_p, xi, a); _, em = flux(along(-1), ab_m, xi, a)
        demu = (ep - em) / (2 * h)
        _, tp = flux(along(+1, True), ab_p, xi, a); _, tm = flux(along(-1, True), ab_m, xi, a)
        dt = (tp - tm) / (2 * h)
        rel = lambda x: float(np.linalg.norm(x - dref) / np.linalg.norm(dref))
        cos = lambda x: float(x @ dref / (np.linalg.norm(x) * np.linalg.norm(dref)))
        # what a fit cares about: the induced label shift error for a residual along the true direction
        gain = float(demu @ dref / (dref @ dref))
        results[lab] = {"rel_l2_emulator": rel(demu), "cosine_emulator": cos(demu), "projected_gain": gain,
                        "rel_l2_T_only_emulator": rel(dt), "rms_dflux_per_unit": float(np.sqrt(np.mean(dref ** 2))),
                        "n_pix": int(wl.size)}
        print(f"{d.name}/{lab}: emulator rel {rel(demu):.3f} cos {cos(demu):.4f} gain {gain:.3f} | "
              f"T-only-from-emulator rel {rel(dt):.3f}", flush=True)
        np.savez(d / f"flux_jac_{lab}.npz", wl=wl, ref=dref, emu=demu, t_only=dt)
    Path(d / "flux_jacobian.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
