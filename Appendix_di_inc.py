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
# 1. 路径配置
# ==========================================
TRUE_DIR = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/model1123_processed_60/Archive/model1123_processed_60/"
RES1_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/model1123_processed_60/output/"
RES2_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/"
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
# ==========================================
# 2. 数据加载 (包含 INC 提取)
# ==========================================
pat_fname = re.compile(
    r"^(?P<id>\d+)_ring_"
    r"v(?P<v>\d+)_R(?P<R>\d+)_"
    r"(?:(?:PAfixed)|PAk(?P<pa_k>-?\d*\.?\d+)_TP(?P<pa_tp>\d+))_"
    r"VR(?:(?:k(?P<vr_k>-?\d*\.?\d+)_TP(?P<vr_tp>\d+))|(?P<vr_fixed>0))_"
    r"INC(?P<inc>\d+)"
)

def load_data_by_inc():
    all_dfs = []
    files = sorted(glob.glob(os.path.join(TRUE_DIR, "*.txt")))
    
    for tf_path in files:
        fname = os.path.basename(tf_path)
        m = pat_fname.match(fname)
        if not m: continue
        
        fid = m.group("id")
        inc_val = int(m.group("inc"))
        
        r1_path = os.path.join(RES1_BASE, fid, "rings_final2.txt")
        folder_name_r2 = fname.replace(f"{fid}_", "").replace(".txt", "")
        r2_path = os.path.join(RES2_BASE, folder_name_r2, "second_step_check_sinw", "rings_final2.txt")
        if os.path.exists(r1_path) and os.path.exists(r2_path):
            # print(tf_path,r1_path,r2_path)
            # try:
            df_t, df_r1, df_r2 = pd.read_csv(tf_path, sep=r"\s+"), pd.read_csv(r1_path,sep= r"\s+"), pd.read_csv(r2_path,sep= r"\s+")
            n = min(len(df_t), len(df_r1), len(df_r2))
            
            combined = pd.DataFrame({
                "True_VRAD": df_t["VRAD(km/s)"][:n].values,
                "R1_VRAD":   df_r1["VRAD(km/s)"][:n].values,
                "R2_VRAD":   df_r2["VRAD(km/s)"][:n].values,
                "True_PA":   df_t["P.A.(deg)"][:n].values,
                "R1_PA":     df_r1["P.A.(deg)"][:n].values,
                "R2_PA":     df_r2["P.A.(deg)"][:n].values,
                "INC":       inc_val
            })
            # 排除 Vrad=0
            combined = combined[combined["True_VRAD"] != 0].reset_index(drop=True)
            if not combined.empty: all_dfs.append(combined)
            # except: continue
                
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# ==========================================
# 3. 绘图函数
# ==========================================
def bin_stats(x, y, nbins=8):
    res = y - x
    valid = ~np.isnan(x) & ~np.isnan(y)
    if not valid.any(): return np.array([]), np.array([]), np.array([])
    
    bin_edges = np.linspace(x[valid].min(), x[valid].max(), nbins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    idx = np.digitize(x, bin_edges) - 1
    means = [res[idx == i].mean() if (idx == i).any() else np.nan for i in range(nbins)]
    stds = [res[idx == i].std() if (idx == i).any() else np.nan for i in range(nbins)]
    return centers, np.array(means), np.array(stds)

def plot_inc_panel(ax_sc, ax_rs, df, t_col, r1_col, r2_col, is_pa=False):
    tx, r1y, r2y = df[t_col].values, df[r1_col].values, df[r2_col].values
    if is_pa: tx, r1y, r2y = tx-120, r1y-120, r2y-120
    
    # Scatter points
    ax_sc.scatter(tx, r1y, s=6, color='dodgerblue', alpha=0.25, label='Di Teodoro & Peek 2021' if not is_pa else "")
    ax_sc.scatter(tx, r2y, s=6, color='darkorange', alpha=0.25, label='This work' if not is_pa else "")
    
    # Trend lines
    for y_data, color in [(r1y, 'dodgerblue'), (r2y, 'darkorange')]:
        c, m, s = bin_stats(tx, y_data)
        if len(c) > 0:
            ax_rs.plot(c, m, color=color, lw=2)
            ax_rs.fill_between(c, m-s, m+s, color=color, alpha=0.15)

    # Styling
    min_val, max_val = tx.min(), tx.max()
    ax_sc.plot([-32, 32], [-32, 32], 'k--', alpha=0.3)
    ax_sc.set_xlim(-32, 32); ax_sc.set_ylim(-32, 32)
    ax_rs.axhline(0, color='k', ls='--', alpha=0.3)
    if is_pa==True:
        ax_rs.set_ylim(-8, 8)
    else:
        ax_rs.set_ylim(-13, 13)
    ax_sc.grid(True, alpha=0.1); ax_rs.grid(True, alpha=0.1)

def main():
    df_all = load_data_by_inc()
    if df_all.empty:
        print("No data loaded. Check paths and regex.")
        return
    
    unique_incs = sorted(df_all["INC"].unique())
    n_incs = len(unique_incs)
    
    fig = plt.figure(figsize=(5 * n_incs, 10))
    # 2 rows (Vrad, PA), n_incs columns
    outer_gs = gridspec.GridSpec(2, n_incs, hspace=0.25, wspace=0.3)
    
    for j, inc in enumerate(unique_incs):
        df_sub = df_all[df_all["INC"] == inc]
        
        # --- Top Row: VRAD ---
        inner_v = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0, j], height_ratios=[3, 1],hspace=0)
        ax_v_sc = fig.add_subplot(inner_v[0])
        ax_v_rs = fig.add_subplot(inner_v[1], sharex=ax_v_sc)
        plot_inc_panel(ax_v_sc, ax_v_rs, df_sub, "True_VRAD", "R1_VRAD", "R2_VRAD")
        ax_v_sc.set_title(f"INC = {inc}°", fontsize=14, fontweight='bold')

        ax_v_sc.legend(loc='upper left', fontsize=10)
        ax_v_sc.set_ylabel(r"Recovered $V_{\rm rad}$ (km/s)", fontsize=12)
        ax_v_rs.set_ylabel("Residual", fontsize=12)
        ax_v_rs.set_xlabel(r"True $V_{\rm rad}$ (km/s)", fontsize=12)
        # --- Bottom Row: PA ---
        inner_p = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[1, j], height_ratios=[3, 1],hspace=0)
        ax_p_sc = fig.add_subplot(inner_p[0])
        ax_p_rs = fig.add_subplot(inner_p[1], sharex=ax_p_sc)
        plot_inc_panel(ax_p_sc, ax_p_rs, df_sub, "True_PA", "R1_PA", "R2_PA", is_pa=True)
        ax_p_sc.set_title(f"INC = {inc}°", fontsize=14, fontweight='bold')
        
        ax_p_sc.set_ylabel("Recovered P.A. (deg)", fontsize=12)
        ax_p_rs.set_ylabel("Residual", fontsize=12)
        ax_p_rs.set_xlabel("True P.A. (deg)", fontsize=12)

    plt.savefig("comparison_by_inclination_DI.png", dpi=300, bbox_inches='tight')
    print("Plot saved as: comparison_by_inclination.png")
    plt.show()

if __name__ == "__main__":
    main()

# DATASERVER3/para_Barolo/test_more_model/model1123_processed_60/Archive/output/06/rings_final2.txt
# DATASERVER3/para_Barolo/test_more_model/model1123/ring_v200_R50_PAfixed_VR0_INC75/second_step_check_sinw