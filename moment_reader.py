#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import cdflib


def print_psp_l3_moments(cdf_path: Path, index: int) -> None:
    cdf = cdflib.CDF(str(cdf_path))

    try:
        dens = cdf["DENS"]
        vel_inst = cdf["VEL_INST"]
        temp = cdf["TEMP"]
    except Exception as e:
        raise RuntimeError(f"Failed to read required variables from CDF: {e}") from e

    n = len(dens)
    if index < 0 or index >= n:
        raise IndexError(f"Index {index} out of range. DENS length is {n}.")

    print(f"KDE match index: {index}")
    print(f"  DENS[{index}]      = {float(dens[index]):.2f} cm^-3")
    print(f"  VEL_INST[{index}]  = {vel_inst[index]} km/s")
    print(f"  TEMP[{index}]      = {float(temp[index]):.1f} eV")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print PSP L3 moments at a given index.")
    parser.add_argument("file", type=Path, help="L3 moments CDF file")
    parser.add_argument("--index", type=int, default=2, help="Record index (default: 2)")
    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f"CDF file not found: {args.file}")

    print_psp_l3_moments(args.file, args.index)


if __name__ == "__main__":
    main()
