from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext

# =========================
# Unit conversion
# =========================

M3_TO_CM3 = 1e-6  # m^-3 → cm^-3


# =========================
# Data loading
# =========================

def load_sampled(sampled_csv: Path) -> np.ndarray:
    """
    Sampled CSV:
      v_perp_0, v_perp_1, v_para

    Returns array with shape (3, N) for gaussian_kde.
    """
    df = pd.read_csv(sampled_csv)

    required = {"v_perp_0", "v_perp_1", "v_para"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in sampled CSV: {missing}")

    vp0 = df["v_perp_0"].to_numpy(float)
    vp1 = df["v_perp_1"].to_numpy(float)
    vpa = df["v_para"].to_numpy(float)

    return np.vstack([vp0, vp1, vpa])


def load_original(original_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Original CSV:
      v_para, v_perp_0, v_perp_1, f(v), counts

    Assumes f(v) is in m^-3 (km/s)^-3 and converts it to
    cm^-3 (km/s)^-3 for KDE scaling.
    """
    df = pd.read_csv(original_csv)

    required = {"v_para", "v_perp_0", "v_perp_1", "f(v)", "counts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in original CSV: {missing}")

    vp0 = df["v_perp_0"].to_numpy(float)
    vp1 = df["v_perp_1"].to_numpy(float)
    vpa = df["v_para"].to_numpy(float)

    # ---- UNIT FIX ----
    f_m3 = df["f(v)"].to_numpy(float)
    f = f_m3 * M3_TO_CM3   # → cm^-3 (km/s)^-3

    K = df["counts"].to_numpy(float)

    V = np.vstack([vp0, vp1, vpa])
    return V, f, K


# =========================
# KDE and scaling
# =========================

def fit_kde(data: np.ndarray, bw_method=None) -> gaussian_kde:
    if data.shape[0] != 3:
        raise ValueError("KDE input must have shape (3, N)")
    return gaussian_kde(data, bw_method=bw_method)


def estimate_n_wls(
    kde: gaussian_kde,
    V: np.ndarray,
    f: np.ndarray,
    K: np.ndarray,
    k_min: float = 0.0,
) -> float:
    """
    Regularized NLS / WLS estimate of n in:

        f_i ≈ n * p(v_i)

    Returns n in cm^-3.
    """
    p = kde(V)

    mask = (
        np.isfinite(f)
        & np.isfinite(K)
        & np.isfinite(p)
        & (K >= k_min)
        & (f > 0.0)
        & (p > 0.0)
    )

    if not np.any(mask):
        raise RuntimeError("No valid bins after filtering for NLS/WLS.")

    f_m = f[mask]
    K_m = K[mask] + 1.0
    p_m = p[mask]

    w = K_m / (f_m ** 2)

    num = np.sum(w * p_m * f_m)
    den = np.sum(w * p_m ** 2)

    n_hat = num / den

    if not np.isfinite(n_hat) or n_hat <= 0.0:
        raise RuntimeError(f"Nonphysical n_hat = {n_hat}")

    return float(n_hat)


def bulk_velocity_from_samples(data: np.ndarray) -> np.ndarray:
    """
    Bulk velocity (normalized first moment) from KDE samples.

    Under a Gaussian KDE, the mean of the KDE equals the sample mean,
    independent of bandwidth.

    Returns a (3,) vector in km/s.
    """
    if data.shape[0] != 3:
        raise ValueError("Sample array must have shape (3, N)")
    return data.mean(axis=1)


# =========================
# Plotting
# =========================

def plot_kde_slices(
    data: np.ndarray,
    kde: gaussian_kde,
    n_scale: float,
    c_values: list[float],
    outpath: Path,
    grid_n: int = 100,
    levels: int = 30,
) -> None:
    vp0 = data[0]
    vp1 = data[1]

    v0_grid, v1_grid = np.mgrid[
        vp0.min():vp0.max():complex(grid_n),
        vp1.min():vp1.max():complex(grid_n),
    ]

    ncols = 3
    nrows = int(np.ceil(len(c_values) / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = axes.ravel()

    contour = None
    for i, c in enumerate(c_values):
        coords = np.vstack([
            v0_grid.ravel(),
            v1_grid.ravel(),
            np.full(v0_grid.size, c),
        ])

        dens = (n_scale * kde(coords)).reshape(v0_grid.shape)

        contour = axes[i].contourf(
            v0_grid,
            v1_grid,
            dens,
            levels=levels,
            cmap="plasma",
        )
        axes[i].set_title(f"v_para = {c}")
        axes[i].set_xlabel("v_perp_0")
        axes[i].set_ylabel("v_perp_1")

    for j in range(len(c_values), axes.size):
        axes[j].axis("off")

    fig.colorbar(
        contour,
        ax=axes[:len(c_values)],
        shrink=0.85,
        label="f(v) [cm$^{-3}$ (km/s)$^{-3}$]",
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KDE VDF reconstruction with regularized NLS/WLS density scaling"
    )

    parser.add_argument("--sampled", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--bw", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("./results"))
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[-375, -225, -75, 75, 225, 375],
    )
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    data = load_sampled(args.sampled)
    kde = fit_kde(data, bw_method=args.bw)

    V, f, K = load_original(args.original)
    n_hat = estimate_n_wls(kde, V, f, K)

    # Bulk velocity vector (km/s) and bulk speed (km/s)
    u_vec = bulk_velocity_from_samples(data)
    u_norm = float(np.linalg.norm(u_vec))

    print("KDE fitted successfully")
    print(f"n_hat [cm^-3] = {n_hat:.6e}")
    print(f"bulk_velocity_vector [km/s] = [{u_vec[0]:.6f}, {u_vec[1]:.6f}, {u_vec[2]:.6f}]")
    print(f"bulk_speed [km/s] = {u_norm:.6f}")

    if args.plot:
        fig_path = args.outdir / f"{args.sampled.stem}_kde_slices_scaled.png"
        plot_kde_slices(
            data=data,
            kde=kde,
            n_scale=n_hat,
            c_values=args.c_values,
            outpath=fig_path,
        )
        print(f"Saved plot to: {fig_path}")


if __name__ == "__main__":
    main()


