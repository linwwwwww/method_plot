#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
})

plt.style.use("/net/dataserver3/data/users/linn/pic_style/science3.mplstyle")
# ==========================================
# 1. 配置与路径 (CONFIG)
# ==========================================
RING_BASE      = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/"
RING_MODEL1123 = RING_BASE + "model1123/"

INITIAL_BASE   = "/net/dataserver3/data/users/linn/para_initial/test_more_model/"
INITIAL_1123   = "/net/dataserver3/data/users/linn/para_initial/test_more_model/model1123/"

# 绘图样式配置
INC_STYLES = {
    45: dict(color="#1f77b4", marker="o", label="$i=45$"),
    60: dict(color="#ff7f0e", marker="s", label="$i=60$"),
    75: dict(color="#2ca02c", marker="^", label="$i=75$"),
}

# ==========================================
# 2. 文件名解析与辅助函数
# ==========================================
pat_fname = re.compile(
    r"^(?:\d+_)?ring_"
    r"v(?P<v>\d+)_R(?P<R>\d+)_"
    r"(?:(?:PAfixed)|PAk(?P<pa_k>-?\d*\.?\d+)_TP(?P<pa_tp>\d+))_"
    r"VR(?:(?:k(?P<vr_k>-?\d*\.?\d+)_TP(?P<vr_tp>\d+))|(?P<vr_fixed>0))_"
    r"INC(?P<inc>\d+)"
)

def compute_sign(pa_k, vr_k):
    if pa_k is None: return 0
    if vr_k is None: return 0
    if vr_k == 0: return 0
    return 1 if pa_k * vr_k > 0 else -1

# ==========================================
# 3. 数据加载函数 (DATA LOADING)
# ==========================================
def load_single_file(file_name, ring_root):
    m = pat_fname.match(file_name)
    if not m: return None

    g = m.groupdict()
    pa_k = float(g["pa_k"]) if g.get("pa_k") else None
    vr_k = float(g["vr_k"]) if g.get("vr_k") else None
    inc_val = int(g["inc"])

    if "model1123" in ring_root:
        in_path = os.path.join(INITIAL_1123, file_name)
    else:
        in_path = os.path.join(INITIAL_BASE, file_name)

    if not os.path.exists(in_path): return None

    try: inputf = pd.read_csv(in_path, delimiter=r"\s+")
    except: return None

    base_name, _ = os.path.splitext(file_name)
    out_path = os.path.join(ring_root, base_name, "second_step_check_sinw", "rings_final2.txt")
    if not os.path.exists(out_path): return None

    try: outputf = pd.read_csv(out_path, delimiter=r"\s+")
    except: return None

    n = min(len(inputf), len(outputf))
    df = pd.DataFrame({
        "True_VRAD": inputf["VRAD(km/s)"][:n].values,
        "Measured_VRAD": outputf["VRAD(km/s)"][:n].values,
        "True_PA": inputf["P.A.(deg)"][:n].values,
        "Measured_PA": outputf["P.A.(deg)"][:n].values,
        "INC": inc_val,
        "pa_k": pa_k,
        "vr_k": vr_k,
        "sign": compute_sign(pa_k, vr_k),
        "file": file_name
    })
    
    df = df[df["True_VRAD"] != 0].reset_index(drop=True)
    return df

def load_all_data():
    df_list = []
    files_1123 = sorted(glob.glob(os.path.join(RING_MODEL1123, "*.txt")))
    for fp in files_1123:
        fname = os.path.basename(fp)
        if "v200" not in fname: continue
        if "INC45" in fname or "INC75" in fname or "INC60" in fname:
            df = load_single_file(fname, RING_MODEL1123)
            if df is not None: df_list.append(df)

    files_base = sorted(glob.glob(os.path.join(RING_BASE, "*.txt")))
    for fp in files_base:
        fname = os.path.basename(fp)
        if "v200" not in fname: continue
        if "INC60" in fname:
            df = load_single_file(fname, RING_BASE)
            if df is not None: df_list.append(df)

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# ==========================================
# 4. 绘图与统计函数 (PLOTTING) - 已更新为 Median/16%/84%
# ==========================================
def bin_stats(x_plot, y_plot, nbins=10):
    """
    计算残差的中位数 (Median) 和 16%/84% 分位数 (取代 Mean/Sigma)
    """
    temp_df = pd.DataFrame({'x': np.array(x_plot), 'y': np.array(y_plot)})
    temp_df['res'] = temp_df['y'] - temp_df['x']

    if temp_df['x'].empty:
        return [], [], [], []
        
    bin_edges = np.linspace(temp_df['x'].min(), temp_df['x'].max(), nbins + 1)
    temp_df['bin_cat'] = pd.cut(temp_df['x'], bins=bin_edges, include_lowest=True)

    grouped = temp_df.groupby('bin_cat', observed=False)['res']
    
    # 获取统计量
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    median_res = grouped.median().values
    q16_res = grouped.quantile(0.16).values
    q84_res = grouped.quantile(0.84).values

    return bin_centers, median_res, q16_res, q84_res

def plot_combined_panel(ax_scatter, ax_resid, x, y, inc_col, label_prefix, is_pa=False, show_legend=False):
    if x.empty or y.empty: return

    for inc, style in INC_STYLES.items():
        sub = inc_col[inc_col == inc].index
        if len(sub) == 0: continue
        
        x_sub, y_sub = x.loc[sub], y.loc[sub]
        x_plot, y_plot = (x_sub - 120, y_sub - 120) if is_pa else (x_sub, y_sub)
            
        # --- Scatter ---
        ax_scatter.scatter(x_plot, y_plot, s=10, alpha=0.6, edgecolor="white", linewidth=0.3, **style,rasterized=True)
        
        # --- Residual (Median + 16/84% Area) ---
        bin_centers, med, q16, q84 = bin_stats(x_plot, y_plot)
        if len(bin_centers) > 0:
            ax_resid.plot(bin_centers, med, **style)
            fill_style = {k: v for k, v in style.items() if k not in ['marker', 'label']}
            ax_resid.fill_between(bin_centers, q16, q84, alpha=0.2, **fill_style)

    # Decorate
    ax_scatter.set_xlim(-32, 32); ax_scatter.set_ylim(-32, 32)
    ax_scatter.plot([-32, 32], [-32, 32], "k--", lw=1, alpha=0.5)
    ax_scatter.grid(True, alpha=0.3); ax_scatter.set_ylabel(f"Recovered {label_prefix}")
    ax_scatter.tick_params(labelbottom=False)
    if show_legend: ax_scatter.legend(loc='lower right', fontsize=14, frameon=True)

    if is_pa:
        ax_resid.set_ylim(-13, 13)
    else:
        ax_resid.set_ylim(-13, 13)
    ax_resid.axhline(0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax_resid.grid(True, alpha=0.3); ax_resid.set_xlabel(f"True {label_prefix}"); ax_resid.set_ylabel("Residuals")

def plot_six_panel_figure(df_all, df_supp, df_enh):
    fig = plt.figure(figsize=(18, 12))
    outer_grid = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.2, wspace=0.25)
    datasets = [("All Data", df_all), ("Suppressing", df_supp), ("Enhancing", df_enh)]

    for row_idx, (lab, is_pa) in enumerate([(r"$V_{\rm rad}$ (km/s)", False), ("P.A. (deg)", True)]):
        for col_idx, (name, df) in enumerate(datasets):
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[row_idx, col_idx], height_ratios=[3, 1], hspace=0.0)
            ax_sc = fig.add_subplot(inner_grid[0])
            ax_rs = fig.add_subplot(inner_grid[1], sharex=ax_sc)
            
            plot_combined_panel(ax_sc, ax_rs, df["True_PA" if is_pa else "True_VRAD"], 
                                df["Measured_PA" if is_pa else "Measured_VRAD"], 
                                df["INC"], label_prefix=lab, is_pa=is_pa, show_legend=(col_idx == 0 and row_idx == 0))
            if not is_pa:
                ax_sc.set_title(f"{name}", fontsize=16, fontweight='bold')

    plt.savefig("figure3-incforincfixedgalaxies.pdf")
    plt.show()

if __name__ == "__main__":
    df_all = load_all_data()
    if not df_all.empty:
        df_supp = df_all[df_all["sign"] > 0].reset_index(drop=True)
        df_enh  = df_all[df_all["sign"] < 0].reset_index(drop=True)
        plot_six_panel_figure(df_all, df_supp, df_enh)
        

        
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 1. 路径与全局配置
# ==========================================
TRUE_DIR = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/snr_res_testing/new_res_output"
RES30_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/snr_res_testing/"
OTHER_RES_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/snr_res_testing/new_res_260128/"

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
plt.style.use("/net/dataserver3/data/users/linn/pic_style/science3.mplstyle")
# 样式配置：使用你要求的配色风格
SNR_STYLES = {
    5:  dict(color="#1f77b4", marker="o", label="SNR 5"),
    10: dict(color="#ff7f0e", marker="s", label="SNR 10"),
    20: dict(color="#2ca02c", marker="^", label="SNR 20"),
}

RES_STYLES = {
    30: dict(color="#1C4D8D", marker="o", label="Res 30"),
    15: dict(color="#79C9C5", marker="s", label="Res 15"),
    7:  dict(color="#2ca02c", marker="^", label="Res 7"),
    4:  dict(color="#F96E5B", marker="D", label="Res 4"),
}

pat_fname = re.compile(
    r"^(?:\d+_)?ring_v(?P<v>\d+)_R(?P<R>\d+)_.*_INC(?P<inc>\d+)_Res(?P<res>\d+)\.txt$"
)

# ==========================================
# 2. 数据处理与统计函数 (Median / 16% / 84%)
# ==========================================
def load_data(base_dir, target_res=None, target_snr=None):
    df_list = []
    files = sorted(glob.glob(os.path.join(base_dir, "*Res*.txt")))
    for fp in files:
        fname = os.path.basename(fp)
        m = pat_fname.match(fname)
        if not m: continue
        res_val = int(m.group("res"))
        if target_res is not None and res_val != target_res: continue
        
        true_path = os.path.join(TRUE_DIR, fname)
        if not os.path.exists(true_path): continue
        df_true = pd.read_csv(true_path, sep='\s+')
        
        snrs = [5, 10, 20] if target_snr is None else [target_snr]
        for snr in snrs:
            folder = fname.replace(".txt", f"_SNR{snr}")
            out_path = os.path.join(base_dir, folder, "second_step_check_sinw", "rings_final2.txt")
            if not os.path.exists(out_path): continue
            
            df_out = pd.read_csv(out_path, sep='\s+')
            n = min(len(df_true), len(df_out))
            df = pd.DataFrame({
                "True_VRAD": df_true["VRAD(km/s)"].values[:n],
                "Measured_VRAD": df_out["VRAD(km/s)"].values[:n],
                "True_PA": df_true["P.A.(deg)"].values[:n],
                "Measured_PA": df_out["P.A.(deg)"].values[:n],
                "SNR": snr, "Res": res_val
            })
            df = df[df["True_VRAD"] != 0].reset_index(drop=True)
            df_list.append(df)
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def bin_stats_median(x, y, cat_val,nbins=10):
    if cat_val == 4:
        nbins = 6
    """计算中位数和 16/84 分位数"""
    df = pd.DataFrame({"x": x, "y": y, "res": y - x})
    if df.empty: return [], [], [], []
    
    bins = np.linspace(df["x"].min(), df["x"].max(), nbins + 1)
    df["bin"] = pd.cut(df["x"], bins=bins, include_lowest=True)
    g = df.groupby("bin", observed=False)["res"]
    
    xc = 0.5 * (bins[:-1] + bins[1:])
    median = g.median().values
    q16 = g.quantile(0.16).values
    q84 = g.quantile(0.84).values
    return xc, median, q16, q84

def plot_panel(ax_sc, ax_rs, df, cat_col, style_dict, label, is_pa=False):
    for cat, style in style_dict.items():
        sub = df[df[cat_col] == cat]
        if sub.empty: continue
        
        xs = (sub["True_PA"] - 120) if is_pa else sub["True_VRAD"]
        ys = (sub["Measured_PA"] - 120) if is_pa else sub["Measured_VRAD"]
        
        # Scatter
        ax_sc.scatter(xs, ys, s=12, alpha=0.5, edgecolor="none", color=style["color"],label=style["label"],rasterized=True)
        
        # Stats: Median + 16/84% area
        xc, med, q16, q84 = bin_stats_median(xs, ys,cat)
        if len(xc) > 0:
            ax_rs.plot(xc, med, color=style["color"], lw=2, marker=style["marker"], ms=6, )
            ax_rs.fill_between(xc, q16, q84, alpha=0.2, color=style["color"], lw=0)

    # Decorate
    ax_sc.plot([-32, 32], [-32, 32], "k--", alpha=0.5, lw=1)
    ax_sc.set_xlim(-32, 32); ax_sc.set_ylim(-32, 32)
    ax_sc.set_ylabel(f"Recovered {label}")
    ax_sc.tick_params(labelbottom=False)
    ax_sc.grid(True, alpha=0.3)
    
    ax_rs.axhline(0, ls="--", color="k", alpha=0.5, lw=1)
    if is_pa:
        ax_rs.set_ylim(-13, 13)
    else:
        ax_rs.set_ylim(-13, 13)
    ax_rs.set_xlabel(f"True {label}")
    ax_rs.set_ylabel("Residuals")
    ax_rs.grid(True, alpha=0.3)

# ==========================================
# 3. 执行绘图 (2x2 布局)
# ==========================================
if __name__ == "__main__":
    # 加载数据
    df_snr_col = load_data(RES30_BASE, target_res=30)
    
    df_res30_s10 = df_snr_col[df_snr_col["SNR"] == 10]
    df_others_s10 = load_data(OTHER_RES_BASE, target_snr=10)
    df_res_col = pd.concat([df_res30_s10, df_others_s10], ignore_index=True)

    fig = plt.figure(figsize=(15, 12))
    outer_gs = gridspec.GridSpec(2, 2, hspace=0.22, wspace=0.28)

    # 定义列的信息：(数据源, 分类字段, 样式字典, 标题)
    cols_info = [
        (df_snr_col, "SNR", SNR_STYLES, "SNR Effects (Res=30)"),
        (df_res_col, "Res", RES_STYLES, "Res Effects (SNR=10)")
    ]

    for col_idx, (data, cat_field, styles, title) in enumerate(cols_info):
        for row_idx, (lab, is_pa) in enumerate([(r"$V_{\rm rad}$ (km/s)", False), ("P.A. (deg)", True)]):
            
            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row_idx, col_idx], 
                                                     height_ratios=[3, 1], hspace=0.0)
            ax_sc = fig.add_subplot(inner[0])
            ax_rs = fig.add_subplot(inner[1], sharex=ax_sc)
            
            plot_panel(ax_sc, ax_rs, data, cat_field, styles, lab, is_pa)
            
            if row_idx == 0:
                ax_sc.set_title(title, fontsize=16, fontweight="bold", pad=20)
                # 将图例放在第一个 row 的残差图中或者 Scatter 图中
                ax_sc.legend(loc="lower right", fontsize=10,  frameon=True)

    plt.savefig("Analysis_SNR_Res_Median_1684.pdf", bbox_inches='tight', dpi=300)
    plt.show()