#!/usr/bin/env python3
"""
analyse_results.py
==================

Читает GSA_DATA.csv и MGGSA_DATA.csv из ./results,
строит графики и формирует сводную таблицу метрик.

Ожидаемые столбцы:
  Iteration, Elapsed_ms, BestFitness, MeanFitness,
  WorstFitness, PopulationDiversity
"""

import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


# ---------- пути ---------- #
ROOT_DIR = pathlib.Path(__file__).parent
RES_DIR  = ROOT_DIR / "results"
GSA_CSV  = RES_DIR / "GSA_DATA.csv"
MG_CSV   = RES_DIR / "MGGSA_DATA.csv"


# ---------- чтение данных ---------- #
def load_data() -> pd.DataFrame:
    if not (GSA_CSV.exists() and MG_CSV.exists()):
        raise FileNotFoundError("Не найден GSA_DATA.csv или MGGSA_DATA.csv в ./results")

    df_gsa = pd.read_csv(GSA_CSV)
    df_gsa["algorithm"] = "GSA"
    df_gsa["runID"] = "GSA_RUN"

    df_mg = pd.read_csv(MG_CSV)
    df_mg["algorithm"] = "MGGSA"
    df_mg["runID"] = "MGGSA_RUN"

    return pd.concat([df_gsa, df_mg], ignore_index=True)


# ---------- утилиты ---------- #
def save_fig(fig: plt.Figure, fname: str):
    fig.tight_layout()
    fig.savefig(RES_DIR / fname, dpi=300)
    plt.close(fig)


def plot_line(df: pd.DataFrame, y_col: str, ylabel: str, log_y: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4))
    for alg, g in df.groupby("algorithm"):
        ax.plot(g["Iteration"], g[y_col], label=alg)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, ls="--", lw=0.5)
    ax.legend()
    save_fig(fig, f"{y_col}_line.png")


def plot_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(
        df["PopulationDiversity"],
        df["BestFitness"],
        c=df["Iteration"],
        cmap="viridis",
        s=10,
        alpha=0.6,
    )
    ax.set_xlabel("Population diversity")
    ax.set_ylabel("Best fitness")
    ax.set_yscale("log")
    ax.grid(True, ls="--", lw=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iteration")
    save_fig(fig, "diversity_vs_bestFitness_scatter.png")


def plot_boxplot(df: pd.DataFrame):
    idx_last = df.groupby("algorithm")["Iteration"].idxmax()
    df_last  = df.loc[idx_last]

    fig, ax = plt.subplots(figsize=(5, 4))
    df_last.boxplot(column="BestFitness", by="algorithm", ax=ax)
    ax.set_ylabel("Final best fitness (log scale)")
    ax.set_yscale("log")
    ax.set_title("Distribution of final best fitness")
    plt.suptitle("")
    ax.grid(axis="y", ls="--", lw=0.5)
    save_fig(fig, "final_fitness_boxplot.png")


# ---------- summary ---------- #
def make_summary(df: pd.DataFrame) -> pathlib.Path:
    idx_last = df.groupby("algorithm")["Iteration"].idxmax()
    df_last  = df.loc[idx_last]

    summary = (
        df_last[["algorithm", "BestFitness", "MeanFitness", "WorstFitness",
                 "PopulationDiversity", "Elapsed_ms"]]
        .rename(
            columns={
                "BestFitness":         "bestFitness_final",
                "MeanFitness":         "meanFitness_final",
                "WorstFitness":        "worstFitness_final",
                "PopulationDiversity": "diversity_final",
                "Elapsed_ms":          "elapsed_ms_final",
            }
        )
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RES_DIR / f"summary_{stamp}.csv"
    summary.to_csv(out_csv, index=False)
    return out_csv


# ---------- основной поток ---------- #
def main():
    RES_DIR.mkdir(exist_ok=True)
    df = load_data()

    # линейные графики
    plot_line(df, "BestFitness", "Best fitness", log_y=True)
    plot_line(df, "MeanFitness", "Mean fitness", log_y=True)
    plot_line(df, "WorstFitness", "Worst fitness", log_y=True)
    plot_line(df, "PopulationDiversity", "Population diversity")
    plot_line(df, "Elapsed_ms", "Iteration time (ms)")

    # scatter и boxplot
    plot_scatter(df)
    plot_boxplot(df)

    # summary
    summary_path = make_summary(df)
    print(f" Графики и таблицы сохранены в {RES_DIR.resolve()}")
    print(f" Сводная таблица: {summary_path.name}")


if __name__ == "__main__":
    main()
