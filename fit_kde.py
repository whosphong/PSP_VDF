from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def load_sampled(sampled_csv: Path) -> np.ndarray:
    df = pd.read_csv(sampled_csv)

    required = {"v_perp_0", "v_perp_1", "v_para"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in sampled CSV: {missing}")

    vp0 = df["v_perp_0"].to_numpy(dtype=float)
    vp1 = df["v_perp_1"].to_numpy(dtype=float)
    vpa = df["v_para"].to_numpy(dtype=float)

    # gaussian_kde expects shape (d, N)
    data = np.vstack([vp0, vp1, vpa])
    return data


def fit_kde(data: np.ndarray, bw_method=None) -> gaussian_kde:
    if data.ndim != 2 or data.shape[0] != 3:
        raise ValueError(f"Expected data shape (3, N). Got {data.shape}")

    kde = gaussian_kde(dataset=data, bw_method=bw_method)
    return kde


def plot_kde_slices(
    data: np.ndarray,
    kde: gaussian_kde,
    c_values: list[float],
    outpath: Path,
    grid_n: int = 100,
    levels: int = 30,
    cmap: str = "plasma",
) -> None:
    vp0 = data[0]
    vp1 = data[1]

    v0_grid, v1_grid = np.mgrid[
        vp0.min():vp0.max():complex(grid_n),
        vp1.min():vp1.max():complex(grid_n),
    ]

    n_slices = len(c_values)
    ncols = 3
    nrows = int(np.ceil(n_slices / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    contour = None
    for i, c in enumerate(c_values):
        coords = np.vstack([
            v0_grid.ravel(),
            v1_grid.ravel(),
            np.full(v0_grid.size, c, dtype=float),
        ])
        dens_slice = kde(coords).reshape(v0_grid.shape)

        ax = axes[i]
        contour = ax.contourf(
            v0_grid,
            v1_grid,
            dens_slice,
            levels=levels,
            cmap=cmap,
        )
        ax.set_title(f"Slice at v_para = {c}")
        ax.set_xlabel("v_perp_0")
        ax.set_ylabel("v_perp_1")

    for j in range(n_slices, axes.size):
        axes[j].axis("off")

    if contour is not None:
        fig.colorbar(
            contour,
            ax=axes[:n_slices],
            orientation="vertical",
            shrink=0.85,
            label="KDE density",
        )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit 3D KDE and save slice plots")
    parser.add_argument(
        "--sampled",
        type=Path,
        required=True,
        help="Path to sampled CSV",
    )
    parser.add_argument(
        "--bw",
        type=float,
        default=None,
        help="Optional bandwidth scale factor",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate KDE slice plots",
    )
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
        help="v_para slice values",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=100,
        help="Grid resolution per axis",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=30,
        help="Number of contour levels",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="plasma",
        help="Matplotlib colormap",
    )

    args = parser.parse_args()

    data = load_sampled(args.sampled)
    kde = fit_kde(data, bw_method=args.bw)

    mu = data.mean(axis=1)
    val = kde(mu)

    print("KDE fitted successfully")
    print(f"Sampled CSV : {args.sampled}")
    print(f"N samples   : {data.shape[1]}")
    print(f"bw_method   : {args.bw if args.bw is not None else 'scipy default'}")
    print(f"p_hat(mean) : {val[0]:.6e}")

    if args.plot:
        args.outdir.mkdir(parents=True, exist_ok=True)
        stem = args.sampled.stem
        fig_path = args.outdir / f"{stem}_kde_slices.png"

        plot_kde_slices(
            data=data,
            kde=kde,
            c_values=args.c_values,
            outpath=fig_path,
            grid_n=args.grid_n,
            levels=args.levels,
            cmap=args.cmap,
        )

        print(f"Saved KDE slice plot to: {fig_path}")


if __name__ == "__main__":
    main()
