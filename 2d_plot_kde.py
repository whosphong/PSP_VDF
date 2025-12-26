from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde


def load_sampled(sampled_csv: Path) -> np.ndarray:
    """
    Sampled CSV columns:
      v_perp_0, v_perp_1, v_para

    Returns:
      data: shape (3, N) with rows [v_perp_0, v_perp_1, v_para]
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
    Original CSV columns:
      v_para, v_perp_0, v_perp_1, f(v), counts

    Returns:
      V: shape (3, M) bin centers [v_perp_0, v_perp_1, v_para]
      f: shape (M,)
      K: shape (M,)
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
    Regularized NLS or WLS estimate of n in:
      f_i ≈ n * p(v_i)
    with:
      K_eff = K + 1
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
        raise RuntimeError("No valid bins after filtering for NLS or WLS.")

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


def bulk_velocity_from_measured(
    V: np.ndarray,
    f: np.ndarray,
    K: np.ndarray,
    k_min: float,
) -> np.ndarray:
    """
    Stable measured bulk velocity proxy using count-weighted mean:
      U ≈ sum K_i v_i / sum K_i
    """
    mask = np.isfinite(V).all(axis=0) & np.isfinite(K) & (K >= k_min)
    if not np.any(mask):
        raise RuntimeError("No valid bins to estimate bulk velocity from measured data.")
    V_m = V[:, mask]
    K_m = K[mask]
    U = (V_m * K_m).sum(axis=1) / K_m.sum()
    return U.astype(float)


def bulk_velocity_from_model(
    kde: gaussian_kde,
    n_mc: int,
    seed: int,
) -> np.ndarray:
    """
    Bulk velocity from the KDE model via resampling:
      U ≈ mean of KDE samples
    """
    np.random.seed(seed)
    samples = kde.resample(n_mc)  # shape (3, n_mc)
    U = samples.mean(axis=1)
    return U.astype(float)


def plot_panel_b_measured_scatter(
    V: np.ndarray,
    f: np.ndarray,
    K: np.ndarray,
    U_meas: np.ndarray,
    outpath: Path,
    up_max: float,
    ua_max: float,
    k_min: float,
    s: float,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    """
    Panel b analogue:
      Scatter of measured bins in (u_perp, u_para) colored by flux f(v) on a log scale.
    """
    mask = (
        np.isfinite(V).all(axis=0)
        & np.isfinite(f)
        & np.isfinite(K)
        & (K >= k_min)
        & (f > 0.0)
    )

    Vc = V[:, mask] - U_meas[:, None]
    u_perp = np.sqrt(Vc[0] ** 2 + Vc[1] ** 2)
    u_para = Vc[2]
    f_m = f[mask]

    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)

    sc = ax.scatter(
        u_perp,
        u_para,
        c=f_m,
        s=s,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidths=0.0,
    )

    ax.set_xlim(0.0, up_max)
    ax.set_ylim(-ua_max, ua_max)

    ax.set_xlabel(r"$v_\perp\;[\mathrm{km/s}]$")
    ax.set_ylabel(r"$v_\parallel\;[\mathrm{km/s}]$")
    ax.set_title("b) Gyrotropic projection (measured, centered)")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$f(v)\;[\mathrm{m^{-3}(km/s)^{-3}}]$")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def gyro_average_kde_on_grid(
    kde: gaussian_kde,
    n_scale: float,
    U_model: np.ndarray,
    up_max: float,
    ua_max: float,
    n_up: int,
    n_ua: int,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the gyrotropic projection of the 3D KDE model on a 2D grid:
      F(up, ua) = (1/n_phi) sum_k n_scale * kde([up cosphi + U0, up sinphi + U1, ua + U2])

    Returns:
      up_grid (n_ua, n_up), ua_grid (n_ua, n_up), F (n_ua, n_up)
    """
    up = np.linspace(0.0, up_max, n_up)
    ua = np.linspace(-ua_max, ua_max, n_ua)
    UP, UA = np.meshgrid(up, ua)

    phis = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    F = np.zeros_like(UP, dtype=float)
    for phi in phis:
        u0 = UP * np.cos(phi)
        u1 = UP * np.sin(phi)
        u2 = UA

        coords = np.vstack(
            [
                (u0.ravel() + U_model[0]),
                (u1.ravel() + U_model[1]),
                (u2.ravel() + U_model[2]),
            ]
        )
        F += (n_scale * kde(coords)).reshape(UP.shape)

    F /= float(n_phi)
    return UP, UA, F


def plot_panel_c_model_contours(
    UP: np.ndarray,
    UA: np.ndarray,
    F: np.ndarray,
    outpath: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    n_levels: int,
) -> None:
    """
    Panel c analogue:
      Filled contours of the gyro-averaged model F(up, ua) with log-spaced levels.
    """
    if np.any(~np.isfinite(F)):
        F = np.where(np.isfinite(F), F, np.nan)

    # Ensure strictly positive for LogNorm and log-spaced levels
    Fp = np.where(F > 0.0, F, np.nan)

    levels = np.logspace(np.log10(vmin), np.log10(vmax), n_levels)

    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)

    cf = ax.contourf(
        UP,
        UA,
        Fp,
        levels=levels,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        extend="both",
    )

    ax.set_xlabel(r"$v_\perp\;[\mathrm{km/s}]$")
    ax.set_ylabel(r"$v_\parallel\;[\mathrm{km/s}]$")
    ax.set_title("c) Gyrotropic projection (KDE reconstruction, centered)")

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$f(v)\;[\mathrm{m^{-3}(km/s)^{-3}}]$")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def robust_flux_limits_for_color(
    f: np.ndarray,
    K: np.ndarray,
    k_min: float,
    lo_q: float,
    hi_q: float,
) -> tuple[float, float]:
    """
    Robust vmin and vmax for log color scaling derived from measured positive flux values.
    """
    mask = np.isfinite(f) & np.isfinite(K) & (K >= k_min) & (f > 0.0)
    if not np.any(mask):
        raise RuntimeError("No valid flux values for color limits.")
    vals = f[mask]
    vmin = float(np.quantile(vals, lo_q))
    vmax = float(np.quantile(vals, hi_q))
    # Prevent degenerate scales
    if vmin <= 0.0:
        vmin = float(np.min(vals[vals > 0.0]))
    if vmax <= vmin:
        vmax = float(np.max(vals))
    return vmin, vmax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce panel b and panel c style gyrotropic figures: measured scatter and KDE reconstruction contours."
    )

    parser.add_argument("--sampled", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("/home/phongth/scratch/results"))
    parser.add_argument("--bw", type=float, default=None, help="KDE bandwidth scale, default scipy")
    parser.add_argument("--kmin-meas", type=float, default=5.0, help="Minimum counts for measured plot and U_meas")
    parser.add_argument("--kmin-color", type=float, default=1.0, help="Minimum counts for flux color limits")

    parser.add_argument("--up-max", type=float, default=500.0, help="Max v_perp axis limit in km/s")
    parser.add_argument("--ua-max", type=float, default=500.0, help="Max abs(v_para) axis limit in km/s")

    parser.add_argument("--grid-up", type=int, default=220, help="u_perp grid points for panel c")
    parser.add_argument("--grid-ua", type=int, default=220, help="u_para grid points for panel c")
    parser.add_argument("--n-phi", type=int, default=96, help="Gyro angles for model averaging")
    parser.add_argument("--mc-U", type=int, default=250_000, help="KDE samples for U_model")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--marker-size", type=float, default=30.0)
    parser.add_argument("--cmap", type=str, default="plasma")

    parser.add_argument("--color-lo-q", type=float, default=0.02, help="Lower quantile for color scale")
    parser.add_argument("--color-hi-q", type=float, default=0.995, help="Upper quantile for color scale")
    parser.add_argument("--levels", type=int, default=9, help="Contour levels for panel c")

    args = parser.parse_args()

    data = load_sampled(args.sampled)
    kde = fit_kde(data, bw_method=args.bw)

    V, f, K = load_original(args.original)
    n_hat = estimate_n_wls(kde, V, f, K)

    U_meas = bulk_velocity_from_measured(V, f, K, k_min=args.kmin_meas)
    U_model = bulk_velocity_from_model(kde, n_mc=args.mc_U, seed=args.seed)

    vmin, vmax = robust_flux_limits_for_color(
        f=f,
        K=K,
        k_min=args.kmin_color,
        lo_q=args.color_lo_q,
        hi_q=args.color_hi_q,
    )

    print("KDE fitted successfully")
    print(f"Sampled CSV : {args.sampled}")
    print(f"Original CSV: {args.original}")
    print(f"N samples   : {data.shape[1]}")
    print(f"bw_method   : {args.bw if args.bw is not None else 'scipy default'}")
    print(f"n_hat (NLS) : {n_hat:.6e}")
    print(f"U_meas      : [{U_meas[0]:.3f}, {U_meas[1]:.3f}, {U_meas[2]:.3f}]")
    print(f"U_model     : [{U_model[0]:.3f}, {U_model[1]:.3f}, {U_model[2]:.3f}]")
    print(f"Color vmin  : {vmin:.6e}")
    print(f"Color vmax  : {vmax:.6e}")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    fig_b = outdir / f"{args.sampled.stem}_panel_b_measured.png"
    plot_panel_b_measured_scatter(
        V=V,
        f=f,
        K=K,
        U_meas=U_meas,
        outpath=fig_b,
        up_max=args.up_max,
        ua_max=args.ua_max,
        k_min=args.kmin_meas,
        s=args.marker_size,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"Saved panel b to: {fig_b}")

    UP, UA, F = gyro_average_kde_on_grid(
        kde=kde,
        n_scale=n_hat,
        U_model=U_model,
        up_max=args.up_max,
        ua_max=args.ua_max,
        n_up=args.grid_up,
        n_ua=args.grid_ua,
        n_phi=args.n_phi,
    )

    fig_c = outdir / f"{args.sampled.stem}_panel_c_kde.png"
    plot_panel_c_model_contours(
        UP=UP,
        UA=UA,
        F=F,
        outpath=fig_c,
        cmap=args.cmap,
        vmin=vmin,
        vmax=vmax,
        n_levels=args.levels,
    )
    print(f"Saved panel c to: {fig_c}")


if __name__ == "__main__":
    main()
