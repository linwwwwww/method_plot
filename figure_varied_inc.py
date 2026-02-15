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
    if not m: return None
    g = m.groupdict()

    inc_base = int(g["inc"]) if g.get("inc") else None
    inc_k    = float(g["inc_k"]) if g.get("inc_k") else 0.0
    inc_tp   = float(g["inc_tp"]) if g.get("inc_tp") else 0.0

    if inc_base == 75 and inc_k < 0:
        if not (np.isclose(inc_k, -0.05) and np.isclose(inc_tp, 450.0)):
            return None

    in_path = os.path.join(INITIAL_DIR, fname)
    if not os.path.exists(in_path): return None

    base, _ = os.path.splitext(fname)
    out_path = os.path.join(RING_MODEL_DIR, base + "_SNR10", "second_step_check_sinw", "rings_final2.txt")
    if not os.path.exists(out_path): return None

    # 使用 sep='\s+' 代替 delim_whitespace
    df_in  = pd.read_csv(in_path, sep='\s+')
    df_out = pd.read_csv(out_path, sep='\s+')

    n = min(len(df_in), len(df_out))
    delta_inc = df_in["INC(deg)"][:n].values - inc_base

    df = pd.DataFrame({
        "Filename": fname,
        "True_VRAD": df_in["VRAD(km/s)"][:n].values,
        "Measured_VRAD": df_out["VRAD(km/s)"][:n].values,
        "True_PA": df_in["P.A.(deg)"][:n].values,
        "Measured_PA": df_out["P.A.(deg)"][:n].values,
        "True_INC": df_in["INC(deg)"][:n].values,
        "Measured_INC": df_out["INC(deg)"][:n].values,
        "Delta_INC": delta_inc,
    })
    return df[df["True_VRAD"] != 0].reset_index(drop=True)

def load_all_data():
    dfs = []
    for fp in sorted(glob.glob(os.path.join(INITIAL_DIR, "*.txt"))):
        fname = os.path.basename(fp)
        if "v200" not in fname: continue
        df = load_single_file(fname)
        if df is not None: dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ==========================================
# 4. BIN STATS (Using Median and 16/84 Percentiles)
# ==========================================
def bin_stats(x, y, nbins=12):
    df = pd.DataFrame({"x": x, "y": y})
    df["res"] = df["y"] - df["x"]
    bins = np.linspace(df["x"].min(), df["x"].max(), nbins + 1)
    df["bin"] = pd.cut(df["x"], bins=bins, include_lowest=True)
    
    g = df.groupby("bin", observed=False)["res"]
    
    xc = 0.5 * (bins[:-1] + bins[1:])
    # 计算中位数, 16% 分位数 和 84% 分位数
    median = g.median().values
    low = g.quantile(0.16).values
    high = g.quantile(0.84).values
    
    return xc, median, low, high

# ==========================================
# 5. PANEL PLOTTER
# ==========================================
def plot_panel(ax_sc, ax_rs, df, xcol, ycol, label, is_pa, cmap, norm, is_inc=False):
    x = df[xcol] - 120 if is_pa else df[xcol]
    y = df[ycol] - 120 if is_pa else df[ycol]

    sc = ax_sc.scatter(x, y, c=df["Delta_INC"], cmap=cmap, norm=norm, s=20, alpha=0.7, edgecolor="none")

    # 获取分位数统计结果
    xc, mid, low, high = bin_stats(x, y)
    
    # 绘制中位数线和 16-84 填充区域
    ax_rs.plot(xc, mid, color="k", lw=2, label="Median")
    ax_rs.fill_between(xc, low, high, color="k", alpha=0.2, label="16th-84th Pctl")

    if is_inc:
        ax_sc.set_xlim(25, 85); ax_sc.set_ylim(25, 85)
        ax_sc.plot([25, 85], [25, 85], "k--", lw=1, alpha=0.5)
    else:
        ax_sc.set_xlim(-30, 30); ax_sc.set_ylim(-30, 30)
        ax_sc.plot([-30, 30], [-30, 30], "k--", lw=1, alpha=0.5)

    ax_sc.grid(alpha=0.3); ax_sc.set_ylabel(f"Recovered {label}")
    ax_sc.tick_params(labelbottom=False)
    
    ax_rs.set_ylim(-15, 15); ax_rs.axhline(0, ls="--", color="k", alpha=0.5)
    ax_rs.grid(alpha=0.3); ax_rs.set_xlabel(f"True {label}"); ax_rs.set_ylabel("Residuals")
    return sc

# ==========================================
# 6. 1×3 ROW FIGURE
# ==========================================
def plot_one_row_all(df_all, title):
    dmax = np.max(np.abs(df_all["Delta_INC"]))
    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
    cmap = plt.get_cmap("RdYlGn")

    fig = plt.figure(figsize=(22, 7))
    fig.suptitle(title, fontsize=16, y=1.02)

    # 1行 4列: Vrad, PA, Inc, Colorbar
    outer = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    rows = [
        ("True_VRAD", "Measured_VRAD", r"$V_{\rm rad}$ (km/s)", False, False),
        ("True_PA",   "Measured_PA",  "P.A. (deg)",        True,  False),
        ("True_INC",  "Measured_INC", "$i$ (deg)",         False, True),
    ]

    mappable = None
    for i, (xcol, ycol, label, is_pa, is_inc) in enumerate(rows):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, i], height_ratios=[3, 1], hspace=0)
        ax_sc = fig.add_subplot(inner[0])
        ax_rs = fig.add_subplot(inner[1], sharex=ax_sc)

        mappable = plot_panel(ax_sc, ax_rs, df_all, xcol, ycol, label, is_pa, cmap, norm, is_inc=is_inc)
        #ax_sc.set_title(f"{label.split()[0]}", fontsize=14, fontweight="bold")

    if mappable is not None:
        cax = fig.add_subplot(outer[0, 3])
        plt.colorbar(mappable, cax=cax).set_label(r"$\Delta i$ (deg)", fontsize=13)

    plt.savefig("figure_varied_inc_vrad_pa_inc_row_median.pdf", bbox_inches='tight')
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