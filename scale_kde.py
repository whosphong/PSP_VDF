from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


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

    Returns:
      V : (3, M) bin centers
      f : (M,) VDF values
      K : (M,) counts
    """
    df = pd.read_csv(original_csv)

    required = {"v_para", "v_perp_0", "v_perp_1", "f(v)", "counts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in original CSV: {missing}")

    vp0 = df["v_perp_0"].to_numpy(float)
    vp1 = df["v_perp_1"].to_numpy(float)
    vpa = df["v_para"].to_numpy(float)
    f = df["f(v)"].to_numpy(float)
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

    using:
        K_eff = K + 1
        sigma_i ≈ f_i / sqrt(K_eff)
        w_i = K_eff / f_i^2
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
    K_m = K[mask] + 1.0   # <<< K ← K + 1 regularization
    p_m = p[mask]

    w = K_m / (f_m ** 2)

    num = np.sum(w * p_m * f_m)
    den = np.sum(w * p_m ** 2)

    n_hat = num / den

    if not np.isfinite(n_hat) or n_hat <= 0.0:
        raise RuntimeError(f"Nonphysical n_hat = {n_hat}")

    return float(n_hat)


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
        label="f(v) = n̂ · KDE",
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

    parser.add_argument(
        "--sampled",
        type=Path,
        required=True,
        help="Sampled CSV (velocities only)",
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Original CSV (bin centers, f(v), counts)",
    )
    parser.add_argument("--bw", type=float, default=None, help="KDE bandwidth scale")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/phongth/scratch/results"),
        help="Output directory",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[-375, -225, -75, 75, 225, 375],
        help="v_para slice locations",
    )
    parser.add_argument("--plot", action="store_true", help="Generate slice plot")

    args = parser.parse_args()

    data = load_sampled(args.sampled)
    kde = fit_kde(data, bw_method=args.bw)

    V, f, K = load_original(args.original)
    n_hat = estimate_n_wls(kde, V, f, K)

    mu = data.mean(axis=1)
    print("KDE fitted successfully")
    print(f"Sampled CSV : {args.sampled}")
    print(f"Original CSV: {args.original}")
    print(f"N samples   : {data.shape[1]}")
    print(f"bw_method   : {args.bw if args.bw is not None else 'scipy default'}")
    print(f"p_hat(mean) : {kde(mu)[0]:.6e}")
    print(f"n_hat (NLS) : {n_hat:.6e}")

    if args.plot:
        fig_path = args.outdir / f"{args.sampled.stem}_kde_slices_scaled_nls.png"
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
