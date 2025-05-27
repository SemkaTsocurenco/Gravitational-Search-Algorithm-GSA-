#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение методов MGGSA и GSA (без стрелок, c легендами).

Файлы-источники:  ./results/MGGSA_runs.csv , ./results/GSA_runs.csv
Графики сохраняются в            ./results/images
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ────────────────────  Пути и чтение данных  ────────────────────
DATA_DIR = "./results"
IMG_DIR  = os.path.join(DATA_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

FILES = {"MGGSA": "MGGSA_runs.csv",
         "GSA":   "GSA_runs.csv"}
dfs = {algo: pd.read_csv(os.path.join(DATA_DIR, name))
       for algo, name in FILES.items()}

OBJ_RU   = {"sphere": "Сфера", "rosenbrock": "Розенброк",
            "rastrigin": "Растригин", "ackley": "Экли"}
OBJ_ORDER = ["sphere", "rosenbrock", "rastrigin", "ackley"]

plt.rcParams.update({
    "figure.figsize": (8, 6),
    "axes.prop_cycle": plt.cycler("color", ["black"]),   # все линии чёрные
    "font.size": 12,
})

# ───────────────────────  1. Box-plot «свечи»  ───────────────────────
fig, ax = plt.subplots()

group_gap = 3.0          # промежуток между функциями
half_gap  = 0.25         # половина ширины пары
base_pos  = np.arange(len(OBJ_ORDER)) * group_gap

for algo, shift, hatch in [("GSA",   -half_gap, "///"),
                           ("MGGSA", +half_gap, "\\\\")]:
    pos = base_pos + shift
    data = [dfs[algo].loc[dfs[algo]["Objective"] == obj, "Iterations"]
            for obj in OBJ_ORDER]

    bp = ax.boxplot(data,
                    positions=pos,
                    widths=half_gap*1.5,
                    patch_artist=True,
                    showfliers=False)          #  ← убрали выбросы

    for patch in bp["boxes"]:
        patch.set(facecolor="white", edgecolor="black", hatch=hatch)

ax.set_xticks(base_pos)
ax.set_xticklabels([OBJ_RU[o] for o in OBJ_ORDER])
ax.set_xlabel("Тестовая функция")
ax.set_ylabel("Кол-во итераций")

# легенда по штриховке
handles = [mpatches.Patch(facecolor='white', edgecolor='black',
                          hatch="///",  label="GSA"),
           mpatches.Patch(facecolor='white', edgecolor='black',
                          hatch="\\\\", label="MGGSA")]
ax.legend(handles=handles, frameon=False, loc="upper right")

fig.tight_layout()
fig.savefig(os.path.join(IMG_DIR, "iterations_box.png"), dpi=300)
plt.close(fig)

# ────────  Вспомогательная функция для двух-линейных графиков  ───────
def dual_line_plot(xcol, xlab, ycol, ylab,
                   file_stub, log_y=False, per_obj=False):

    targets = OBJ_ORDER if per_obj else [None]
    for obj in targets:
        fig, ax = plt.subplots()

        for algo, style in [("MGGSA", "-"), ("GSA", "--")]:
            df = dfs[algo]
            if obj:
                df = df[df["Objective"] == obj]

            grp = df.groupby(xcol)[ycol].median().reset_index()
            ax.plot(grp[xcol], grp[ycol],
                    linestyle=style, marker="o", linewidth=1,
                    label=algo)

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        title = f"{OBJ_RU[obj]}: {ylab} vs {xlab.lower()}" if obj else f"{ylab} vs {xlab.lower()}"
        ax.set_title(title)

        ax.legend(frameon=False, loc="best")

        fname = f"{file_stub}_{obj}.png" if obj else f"{file_stub}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(IMG_DIR, fname), dpi=300)
        plt.close(fig)

# ─────────────────── 2. Точность ⟂ N  (по каждой функции) ──────────────────
dual_line_plot("N", "Количество частиц",
               "BestFitness", "Точность (BestFitness)",
               file_stub="bestfitness_vs_N", log_y=True, per_obj=True)

# ─────────────────── 3. Точность ⟂ D  (по каждой функции) ──────────────────
dual_line_plot("D", "Размерность задачи",
               "BestFitness", "Точность (BestFitness)",
               file_stub="bestfitness_vs_D", log_y=True, per_obj=True)

# ──────────────── 4. Время выполнения ⟂ D  (общий график) ────────────────
dual_line_plot("D", "Размерность задачи",
               "Elapsed_ms", "Время выполнения (мс)",
               file_stub="runtime_vs_D", log_y=False, per_obj=False)

# ──────────────── 5. Время выполнения ⟂ N  (общий график) ────────────────
dual_line_plot("N", "Количество частиц",
               "Elapsed_ms", "Время выполнения (мс)",
               file_stub="runtime_vs_N", log_y=False, per_obj=False)

print(" Графики пересохранены в:", IMG_DIR)
