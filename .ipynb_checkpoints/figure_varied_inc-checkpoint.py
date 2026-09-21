#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ==========================================
# 1. PATH CONFIG
# ==========================================
plt.style.use("/net/dataserver3/data/users/linn/pic_style/science3.mplstyle")

RING_MODEL_DIR = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/varied_inc/"
INITIAL_DIR    = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/varied_inc/initial/"


# ==========================================
# 2. FILENAME PARSER
# ==========================================
pat_fname = re.compile(
    r"^(?:\d+_)?ring_"
    r"v(?P<v>\d+)_R(?P<R>\d+)_"
    r"(?:(?:PAfixed)|PAk(?P<pa_k>-?\d*\.?\d+)_TP(?P<pa_tp>-?\d*\.?\d+))_"
    r"VR(?:(?:k(?P<vr_k>-?\d*\.?\d+)_TP(?P<vr_tp>-?\d*\.?\d+))|(?P<vr_fixed>0))_"
    r"INC(?P<inc>\d+)(?:k(?P<inc_k>-?\d*\.?\d+)_TP(?P<inc_tp>-?\d*\.?\d+))?"
    r"(?:_Res\d+)?"
)


# ==========================================
# 3. DATA LOADING
# ==========================================
def load_single_file(fname):

    m = pat_fname.match(fname)
    if not m:
        return None

    g = m.groupdict()

    inc_base = int(g["inc"]) if g.get("inc") else None
    inc_k    = float(g["inc_k"]) if g.get("inc_k") else 0.0
    inc_tp   = float(g["inc_tp"]) if g.get("inc_tp") else 0.0

    if inc_base == 75 and inc_k < 0:
        if not (np.isclose(inc_k, -0.05) and np.isclose(inc_tp, 450.0)):
            return None

    in_path = os.path.join(INITIAL_DIR, fname)
    if not os.path.exists(in_path):
        return None

    base, _ = os.path.splitext(fname)

    out_path = os.path.join(
        RING_MODEL_DIR,
        base + "_SNR10",
        "second_step_check_sinw",
        "rings_final2.txt"
    )

    if not os.path.exists(out_path):
        return None

    df_in  = pd.read_csv(in_path,  sep='\s+')
    df_out = pd.read_csv(out_path, sep='\s+')

    n = min(len(df_in), len(df_out))

    delta_inc = df_in["INC(deg)"][:n].values - inc_base
    measured_delta_inc = df_out["INC(deg)"][:n].values - inc_base

    df = pd.DataFrame({

        "Filename": fname,

        "Base_INC": inc_base,

        "True_VRAD": df_in["VRAD(km/s)"][:n].values,
        "Measured_VRAD": df_out["VRAD(km/s)"][:n].values,

        "True_PA": df_in["P.A.(deg)"][:n].values,
        "Measured_PA": df_out["P.A.(deg)"][:n].values,

        "True_INC": df_in["INC(deg)"][:n].values,
        "Measured_INC": df_out["INC(deg)"][:n].values,

        "Delta_INC": delta_inc,
        "Measured_delta_inc": measured_delta_inc
    })

    return df[df["True_VRAD"] != 0].reset_index(drop=True)


def load_all_data():

    dfs = []

    for fp in sorted(glob.glob(os.path.join(INITIAL_DIR, "*.txt"))):

        fname = os.path.basename(fp)

        if "v200" not in fname:
            continue

        df = load_single_file(fname)

        if df is not None:
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ==========================================
# 4. BIN STATS
# ==========================================
def bin_stats(x, y, nbins=12):

    df = pd.DataFrame({"x": x, "y": y})

    df["res"] = df["y"] - df["x"]

    bins = np.linspace(df["x"].min(), df["x"].max(), nbins + 1)

    df["bin"] = pd.cut(df["x"], bins=bins, include_lowest=True)

    g = df.groupby("bin", observed=False)["res"]

    xc = 0.5 * (bins[:-1] + bins[1:])

    median = g.median().values
    low    = g.quantile(0.16).values
    high   = g.quantile(0.84).values

    return xc, median, low, high


# ==========================================
# 5. PANEL PLOTTER
# ==========================================
def plot_panel(ax_sc, ax_rs, df, xcol, ycol, label, is_pa, cmap, norm, color_values, is_inc=False):

    x = df[xcol] - 120 if is_pa else df[xcol]
    y = df[ycol] - 120 if is_pa else df[ycol]

    sc = ax_sc.scatter(
        x,
        y,
        c=color_values,
        cmap=cmap,
        norm=norm,
        s=20,
        alpha=0.7,
        edgecolor="none"
    )

    xc, mid, low, high = bin_stats(x, y)

    ax_rs.plot(xc, mid, color="k", lw=2)
    ax_rs.fill_between(xc, low, high, color="k", alpha=0.2)

    ax_sc.set_xlim(-30, 30)
    ax_sc.set_ylim(-30, 30)

    ax_sc.plot([-30, 30], [-30, 30], "k--", lw=1, alpha=0.5)

    ax_sc.grid(alpha=0.3)
    ax_sc.set_ylabel(f"Recovered {label}")
    ax_sc.tick_params(labelbottom=False)

    ax_rs.set_ylim(-15, 15)
    ax_rs.axhline(0, ls="--", color="k", alpha=0.5)
    ax_rs.grid(alpha=0.3)

    ax_rs.set_xlabel(f"True {label}")
    ax_rs.set_ylabel("Residuals")

    return sc


# ==========================================
# 6. FIGURE
# ==========================================
def plot_one_row_all(df_all, title):

    dmax = np.max(np.abs(df_all["Delta_INC"]))

    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    cmap = plt.get_cmap("RdYlGn")

    fig = plt.figure(figsize=(22, 7))

    fig.suptitle(title, fontsize=16, y=1.02)

    outer = gridspec.GridSpec(1, 4, width_ratios=[0.05, 1, 1, 1], wspace=0.3)

    rows = [

        ("True_VRAD", "Measured_VRAD", r"$V_{\rm rad}$ (km/s)", False, False),

        ("True_PA",   "Measured_PA",  "P.A. (deg)", True, False),

        ("Delta_INC", "Measured_delta_inc", "$i$ (deg)", False, True)

    ]

    mappable = None

    for i, (xcol, ycol, label, is_pa, is_inc) in enumerate(rows):

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[0, i+1],
            height_ratios=[3, 1],
            hspace=0
        )

        ax_sc = fig.add_subplot(inner[0])
        ax_rs = fig.add_subplot(inner[1], sharex=ax_sc)

        # 左和中 panel
        if i < 2:

            color_values = df_all["Delta_INC"]

            mappable = plot_panel(
                ax_sc, ax_rs,
                df_all,
                xcol, ycol,
                label,
                is_pa,
                cmap,
                norm,
                color_values,
                is_inc=is_inc
            )

        # 右 panel
        else:

            colors = {

                45: "#1f77b4",
                60: "#ff7f0e",
                75: "#2ca02c"

            }

            for inc in [45, 60, 75]:
            
                sub = df_all[df_all["Base_INC"] == inc]
            
                x = sub[xcol]
                y = sub[ycol]
            
                ax_sc.scatter(
                    x,
                    y,
                    color=colors[inc],
                    s=10,
                    alpha=0.6,
                    edgecolor="white",
                    linewidth=0.3,
                    label=f"$i={inc}$°",rasterized=True
                )
            
                # 各自的 residual fit
                xc, mid, low, high = bin_stats(x, y)
            
                ax_rs.plot(
                    xc,
                    mid,
                    color=colors[inc],
                    lw=2
                )
            
                ax_rs.fill_between(
                    xc,
                    low,
                    high,
                    color=colors[inc],
                    alpha=0.2
                )

            # xc, mid, low, high = bin_stats(df_all[xcol], df_all[ycol])

            # ax_rs.plot(xc, mid, color="k", lw=2)
            # ax_rs.fill_between(xc, low, high, color="k", alpha=0.2)
            
            ax_sc.set_xlim(-30, 30)
            ax_sc.set_ylim(-30, 30)
            ax_sc.plot([-30, 30], [-30, 30], "k--", lw=1, alpha=0.5)

            if is_inc:
                ax_sc.set_xlim(-20, 20)
                ax_sc.set_ylim(-20, 20)
                ax_sc.plot([-20, 20], [-20, 20], "k--", lw=1, alpha=0.5)
                

            ax_sc.grid(alpha=0.3)
            ax_sc.set_ylabel(f"Recovered {label}")
            ax_sc.tick_params(labelbottom=False)

            ax_rs.set_ylim(-15, 15)
            ax_rs.axhline(0, ls="--", color="k", alpha=0.5)
            ax_rs.grid(alpha=0.3)

            ax_rs.set_xlabel(f"True {label}")
            ax_rs.set_ylabel("Residuals")

            ax_sc.legend(title="Base $i$",frameon=True,fontsize=12)

    if mappable is not None:

        cax = fig.add_subplot(outer[0, 0])

        cbar = plt.colorbar(mappable, cax=cax)
        cbar.set_label(r"$\Delta i$ (deg)", fontsize=13)
        
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("left")

    plt.savefig(
        "figure_varied_inc_vrad_pa_inc_row_median_inc_new.pdf",
        bbox_inches='tight'
    )

    plt.show()


# ==========================================
# 7. MAIN
# ==========================================
if __name__ == "__main__":

    print("Loading data...")

    df_all = load_all_data()

    if not df_all.empty:

        print(f"Total points: {len(df_all)}")

        plot_one_row_all(df_all, "")

    else:

        print("Data empty.")