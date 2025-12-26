from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_vdf(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    required = {"v_perp_0", "v_perp_1", "v_para", "f(v)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def sample_from_vdf(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    vp0 = df["v_perp_0"].to_numpy()
    vp1 = df["v_perp_1"].to_numpy()
    vpa = df["v_para"].to_numpy()
    fval = df["f(v)"].to_numpy(dtype=float)

    # Avoid zero-weight bins
    fval += 1.0

    # Normalize to PMF
    p = fval / np.sum(fval)

    # Expected counts
    counts = np.round(n_samples * p).astype(int)

    # Enforce exact total
    diff = n_samples - counts.sum()
    if diff != 0:
        order = np.argsort(-p)
        i = 0
        while diff != 0:
            idx = order[i % len(order)]
            if diff > 0:
                counts[idx] += 1
                diff -= 1
            elif counts[idx] > 0:
                counts[idx] -= 1
                diff += 1
            i += 1

    sampled_df = pd.DataFrame(
        {
            "v_perp_0": np.repeat(vp0, counts),
            "v_perp_1": np.repeat(vp1, counts),
            "v_para": np.repeat(vpa, counts),
        }
    )

    return sampled_df



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample synthetic particles from a PSP VDF CSV."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input VDF CSV file",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write sampled CSV",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of synthetic samples to generate",
    )

    return parser.parse_args()



def main() -> None:
    args = parse_args()

    input_csv: Path = args.input
    output_dir: Path = args.output_dir
    n_samples: int = args.n_samples

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_vdf(input_csv)
    sampled_df = sample_from_vdf(df, n_samples)

    output_name = input_csv.stem + "_sampled.csv"
    output_path = output_dir / output_name

    sampled_df.to_csv(output_path, index=False)

    print("Sampling complete")
    print(f"Input file : {input_csv}")
    print(f"Output file: {output_path}")
    print(f"Samples    : {len(sampled_df)}")


if __name__ == "__main__":
    main()
