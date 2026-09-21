#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("/net/dataserver3/data/users/linn/pic_style/science3.mplstyle")
# ==========================================
# 1. Path Configuration
# ==========================================
TRUE_DIR = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/model1123_processed_60/Archive/model1123_processed_60/"
RES1_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/model1123_processed_60/output/"
RES2_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/"

# ==========================================
# 2. Regex & Plot Settings
# ==========================================
pat_fname = re.compile(
    r"^(?P<id>\d+)_ring_"
    r"v(?P<v>\d+)_R(?P<R>\d+)_"
    r"(?:(?:PAfixed)|PAk(?P<pa_k>-?\d*\.?\d+)_TP(?P<pa_tp>\d+))_"
    r"VR(?:(?:k(?P<vr_k>-?\d*\.?\d+)_TP(?P<vr_tp>\d+))|(?P<vr_fixed>0))_"
    r"INC(?P<inc>\d+)"
)

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ==========================================
# 3. Data Loading Logic
# ==========================================
def load_data():
    all_dfs = []
    files = sorted(glob.glob(os.path.join(TRUE_DIR, "*.txt")))
    
    for tf_path in files:
        fname = os.path.basename(tf_path)
        m = pat_fname.match(fname)
        if not m: continue
        
        fid = m.group("id")
        r1_path = os.path.join(RES1_BASE, fid, "rings_final2.txt")
        
        folder_name_r2 = fname.replace(f"{fid}_", "").replace(".txt", "")
        r2_path = os.path.join(RES2_BASE, folder_name_r2, "second_step_check_sinw", "rings_final2.txt")
        
        if os.path.exists(r1_path) and os.path.exists(r2_path):
            try:
                df_t  = pd.read_csv(tf_path, delimiter=r"\s+")
                df_r1 = pd.read_csv(r1_path, delimiter=r"\s+")
                df_r2 = pd.read_csv(r2_path, delimiter=r"\s+")
                
                n = min(len(df_t), len(df_r1), len(df_r2))
                
                combined = pd.DataFrame({
                    "True_VRAD": df_t["VRAD(km/s)"][:n].values,
                    "R1_VRAD":   df_r1["VRAD(km/s)"][:n].values,
                    "R2_VRAD":   df_r2["VRAD(km/s)"][:n].values,
                    "True_PA":   df_t["P.A.(deg)"][:n].values,
                    "R1_PA":      df_r1["P.A.(deg)"][:n].values,
                    "R2_PA":      df_r2["P.A.(deg)"][:n].values,
                })
                
                combined = combined[combined["True_VRAD"] != 0].reset_index(drop=True)
                
                if not combined.empty:
                    all_dfs.append(combined)
            except Exception as e:
                print(f"Error loading {fid}: {e}")
                
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ==========================================
# 4. Statistical Functions
# ==========================================
def bin_stats_robust(x, y, nbins=10):
    res = y - x
    bin_edges = np.linspace(-30, 30, nbins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    idx = np.digitize(x, bin_edges) - 1
    
    medians, p16, p84 = [], [], []
    for i in range(nbins):
        mask = (idx == i)
        if mask.any():
            medians.append(np.nanpercentile(res[mask], 50))
            p16.append(np.nanpercentile(res[mask], 16))
            p84.append(np.nanpercentile(res[mask], 84))
        else:
            medians.append(np.nan); p16.append(np.nan); p84.append(np.nan)
            
    return centers, np.array(medians), np.array(p16), np.array(p84)

def print_summary_stats(df):
    """计算并打印全局残差的统计信息"""
    tasks = [
        ("Vrad (km/s)", "True_VRAD", "R1_VRAD", "R2_VRAD"),
        ("P.A. (deg)",  "True_PA",   "R1_PA",   "R2_PA")
    ]
    
    print("\n" + "="*50)
    print(f"{'Parameter':<12} | {'Method':<25} | {'Median':>8} | {'16th':>8} | {'84th':>8}")
    print("-" * 50)
    
    for label, t_col, r1_col, r2_col in tasks:
        # 残差 = 恢复值 - 真实值
        res1 = df[r1_col] - df[t_col]
        res2 = df[r2_col] - df[t_col]
        
        for m_name, res in [("Di Teodoro & Peek 2021", res1), ("This Work", res2)]:
            med = np.nanpercentile(res, 50)
            p16 = np.nanpercentile(res, 16)
            p84 = np.nanpercentile(res, 84)
            print(f"{label:<12} | {m_name:<25} | {med:8.3f} | {p16:8.3f} | {p84:8.3f}")
        print("-" * 50)

# ==========================================
# 5. Plotting Function
# ==========================================
def plot_comparison(df):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.0, wspace=0.25)
    
    tasks = [
        (r"$V_{\rm rad}$ (km/s)", "True_VRAD", "R1_VRAD", "R2_VRAD", False, 0),
        ("P.A. (deg)",      "True_PA",   "R1_PA",   "R2_PA",   True,  1)
    ]
    
    for label, t_col, r1_col, r2_col, is_pa, col_idx in tasks:
        ax_sc = fig.add_subplot(gs[0, col_idx])
        ax_rs = fig.add_subplot(gs[1, col_idx], sharex=ax_sc)
        
        tx = df[t_col].values
        r1y = df[r1_col].values
        r2y = df[r2_col].values
        
        if is_pa:
            tx, r1y, r2y = tx - 120, r1y - 120, r2y - 120
        
        ax_sc.scatter(tx, r1y, s=10, color='dodgerblue', alpha=0.2, label='Di teodoro & Peek 2021', zorder=2,rasterized=True)
        ax_sc.scatter(tx, r2y, s=10, color='darkorange', alpha=0.2, label='This work', zorder=3,rasterized=True)
        
        for y_data, color in [(r1y, 'dodgerblue'), (r2y, 'darkorange')]:
            c, med, low, high = bin_stats_robust(tx, y_data)
            ax_rs.plot(c, med, color=color, lw=2.5, zorder=5)
            ax_rs.fill_between(c, low, high, color=color, alpha=0.15, lw=0)

        ax_sc.set_box_aspect(1)
        ax_rs.set_box_aspect(1/3)
        ax_sc.set_anchor('S')
        ax_rs.set_anchor('N')
        
        plt.setp(ax_sc.get_xticklabels(), visible=False)

        ax_sc.plot([-30, 30], [-30, 30], 'k--', alpha=0.6, zorder=1)
        ax_sc.set_ylabel(f"Recovered {label}")
        
        if not is_pa:
            ax_sc.legend(loc='upper left', frameon=True, fontsize=16)
            
        ax_rs.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_rs.set_ylabel("Residuals")
        ax_rs.set_xlabel(f"True {label}")
        
        ax_sc.set_xlim(-30, 30)
        ax_sc.set_ylim(-30, 30)
        ax_rs.set_ylim(-15, 15)
        ax_sc.grid(True, alpha=0.15)
        ax_rs.grid(True, alpha=0.15)

    plt.savefig("vrad_pa_comparison_snapped.pdf", bbox_inches='tight')
    plt.show()

# ==========================================
# 6. Execution
# ==========================================
if __name__ == "__main__":
    data = load_data()
    if not data.empty:
        print(f"Total valid data points: {len(data)}")
        # 1. 打印统计数值
        print_summary_stats(data)
        # 2. 绘图
        plot_comparison(data)
    else:
        print("Error: No data found. Please verify the folder paths.")