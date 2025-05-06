#!/usr/bin/env python3
"""
plot_convergence.py – визуализация сходимости GSA vs MGGSA
----------------------------------------------------------
usage:
    python plot_convergence.py              # X = iteration
    python plot_convergence.py --x time     # X = elapsed ms
"""
import argparse
import pathlib

import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent / "./results"

def load_csv(name: str) -> pd.DataFrame:
    path = ROOT / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if {"iteration", "time_ms", "best_fit"} <= set(df.columns):
        return df[["iteration", "time_ms", "best_fit"]]
    # fallback: первый, второй, третий столбец
    df.columns = ["iteration", "time_ms", "best_fit"]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", choices=["iter", "time"], default="iter",
                        help="ось X: iter=номер итерации (по умолч.), time=мс")
    args = parser.parse_args()

    gsa   = load_csv("GSA_DATA.csv").rename(columns={"best_fit": "gsa_bf"})
    mggsa = load_csv("MGGSA_DATA.csv").rename(columns={"best_fit": "mggsa_bf"})

    # выравниваем длину: inner join по итерациям или времени
    key = "iteration" if args.x == "iter" else "time_ms"
    df = pd.merge(gsa[[key, "gsa_bf"]],
                  mggsa[[key, "mggsa_bf"]],
                  on=key, how="outer").sort_values(key)
    df.interpolate(method="pad", inplace=True)   # заполняем NaN «предыдущим»

    plt.figure()
    plt.semilogy(df[key], df["gsa_bf"],   label="GSA")
    plt.semilogy(df[key], df["mggsa_bf"], label="MGGSA")
    plt.xlabel("iteration" if args.x == "iter" else "elapsed time, ms")
    plt.ylabel("best fitness (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(ROOT / "GSA_Fitnes")
    plt.show()

if __name__ == "__main__":
    main()
