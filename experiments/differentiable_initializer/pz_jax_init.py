"""JAX port of Payne Zero's five-label atmosphere initializer (differentiable in the labels).

Mirrors payne_zero_atmosphere.warm_start.AtmosphereInitializer.predict, minus the text-deck
round trip and the clips (inactive inside the training support), so jax.jacfwd works.

    python pz_jax_init.py export     # checkpoint.pt -> work/five_label.npz (once)
    python pz_jax_init.py parity     # JAX port vs the Torch/NumPy original at test labels
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = Path(os.environ.get("DIFFINIT_WORK", HERE / "work"))
RESULTS = HERE / "results"
CHECKPOINT = REPO / "source_data_files" / "atmosphere_emulator" / "five_label" / "checkpoint.pt"
WEIGHTS = WORK / "five_label.npz"

FIELDS = ("column_mass", "temperature", "gas_pressure", "electron_density",
          "rosseland_opacity", "radiative_acceleration")


def export_checkpoint(pt_path: Path = CHECKPOINT, npz_path: Path = WEIGHTS) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arrays = {
        "label_mean": np.asarray(ck["labels"]["mean"], np.float64),
        "label_std": np.asarray(ck["labels"]["std"], np.float64),
        "label_bounds": np.asarray([ck["labels"]["bounds"][f] for f in ck["labels"]["fields"]], np.float64),
        "tau": np.asarray(ck["coordinates"]["standard_rosseland_optical_depth"], np.float64),
        "acceleration_scale": np.float64(ck["coordinates"]["acceleration_scale"]),
    }
    for k in ("coordinate_mean", "coordinate_std", "basis", "coefficient_mean", "coefficient_std"):
        arrays["pca_" + k] = np.asarray(ck["pca"][k], np.float64)
    sd = ck["model"]["state_dict"]
    layer_ids = sorted({int(k.split(".")[0]) for k in sd})
    for i, lid in enumerate(layer_ids):
        arrays[f"W{i}"] = sd[f"{lid}.weight"].double().numpy()
        arrays[f"b{i}"] = sd[f"{lid}.bias"].double().numpy()
    arrays["n_layers"] = np.int64(len(layer_ids))
    np.savez(npz_path, **arrays)


class JaxInitializer:
    def __init__(self, npz_path: Path = WEIGHTS, mlp_dtype=jnp.float32):
        z = np.load(npz_path)
        self.p = {k: jnp.asarray(z[k]) for k in z.files if k != "n_layers"}
        self.n_layers = int(z["n_layers"])
        self.mlp_dtype = mlp_dtype  # torch ran the MLP in float32
        self.bounds = np.asarray(z["label_bounds"])

    def coordinates(self, labels):
        """labels = (Teff, logg, [M/H], [alpha/M], xi_km_s) -> (80, 6) decoded fields."""
        p = self.p
        teff, logg, mh, am, xi = labels[0], labels[1], labels[2], labels[3], labels[4]
        feats = jnp.stack([5040.0 / teff, logg, mh, am, xi])
        x = ((feats - p["label_mean"]) / p["label_std"]).astype(self.mlp_dtype)
        for i in range(self.n_layers):
            x = x @ p[f"W{i}"].T.astype(self.mlp_dtype) + p[f"b{i}"].astype(self.mlp_dtype)
            if i < self.n_layers - 1:
                x = jax.nn.silu(x)
        coef = x.astype(jnp.float64) * p["pca_coefficient_std"] + p["pca_coefficient_mean"]
        flat = (coef @ p["pca_basis"]) * p["pca_coordinate_std"] + p["pca_coordinate_mean"]
        c = flat.reshape(80, 6)
        grey = teff * (0.75 * (p["tau"] + 2.0 / 3.0)) ** 0.25
        return jnp.stack([
            jnp.cumsum(10.0 ** c[:, 0]),
            grey * 10.0 ** c[:, 1],
            10.0 ** c[:, 2],
            10.0 ** c[:, 3],
            10.0 ** c[:, 4],
            p["acceleration_scale"] * jnp.sinh(c[:, 5]),
        ], axis=1)

    def dss_state(self, labels):
        """(4, 80) T, P_gas, m, n_e -- the array dss.couple.atmosphere.state_fn returns."""
        d = self.coordinates(labels)
        return jnp.stack([d[:, 1], d[:, 2], d[:, 0], d[:, 3]])

    def log_state_jacobian(self, labels):
        """d ln(T, P_gas, m, n_e) / d (Teff, logg, [M/H], [alpha/M], xi): shape (4, 80, 5)."""
        return jax.jacfwd(lambda l: jnp.log(self.dss_state(l)))(jnp.asarray(labels, jnp.float64))


PARITY_LABELS = [(5777, 4.44, 0.0, 0.0, 1.0), (4286, 1.66, -0.52, 0.30, 1.7),
                 (5000, 2.50, -1.50, 0.40, 1.5), (9000, 4.2, 0.3, 0.0, 2.0), (4100, 1.0, -2.0, 0.4, 2.5)]


def parity() -> dict:
    """Max relative difference per field, JAX port vs payne_zero_atmosphere's own predict()."""
    from payne_zero_atmosphere.warm_start import load_atmosphere_initializer
    ref = load_atmosphere_initializer(checkpoint_path=CHECKPOINT, device="cpu")
    f = jax.jit(JaxInitializer().coordinates)
    out = {}
    for lab in PARITY_LABELS:
        r = ref.predict(effective_temperature=lab[0], log_surface_gravity=lab[1], metallicity=lab[2],
                        alpha_enhancement=lab[3], microturbulence_km_s=lab[4])
        y = np.asarray(f(jnp.array(lab, float)))
        errs = {}
        for i, k in enumerate(FIELDS):
            if k == "radiative_acceleration":    # signed field: scale by its maximum magnitude
                errs[k] = float(np.max(np.abs(y[:, i] - r[k])) / np.max(np.abs(r[k])))
            else:
                errs[k] = float(np.max(np.abs(y[:, i] / r[k] - 1.0)))
        out[str(lab)] = errs
        print(lab, "  ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("export", "parity"))
    a = p.parse_args()
    if a.cmd == "export":
        export_checkpoint()
        print(f"wrote {WEIGHTS}")
    else:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "parity.json").write_text(json.dumps(parity(), indent=1))
