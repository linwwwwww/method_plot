#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# ==========================================
# 1. 配置与路径 (CONFIG)
# ==========================================
RING_BASE      = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/"
RING_MODEL1123 = RING_BASE + "model1123/"

INITIAL_BASE   = "/net/dataserver3/data/users/linn/para_initial/test_more_model/"
INITIAL_1123   = "/net/dataserver3/data/users/linn/para_initial/test_more_model/model1123/"

# 绘图样式配置
INC_STYLES = {
    45: dict(color="#1f77b4", marker="o", label="INC45"),
    60: dict(color="#ff7f0e", marker="s", label="INC60"),
    75: dict(color="#2ca02c", marker="^", label="INC75"),
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
        ax_scatter.scatter(x_plot, y_plot, s=10, alpha=0.6, edgecolor="white", linewidth=0.3, **style)
        
        # --- Residual (Median + 16/84% Area) ---
        bin_centers, med, q16, q84 = bin_stats(x_plot, y_plot)
        if len(bin_centers) > 0:
            ax_resid.plot(bin_centers, med, **style)
            fill_style = {k: v for k, v in style.items() if k not in ['marker', 'label']}
            ax_resid.fill_between(bin_centers, q16, q84, alpha=0.2, **fill_style)

    # Decorate
    ax_scatter.set_xlim(-30, 30); ax_scatter.set_ylim(-30, 30)
    ax_scatter.plot([-30, 30], [-30, 30], "k--", lw=1, alpha=0.5)
    ax_scatter.grid(True, alpha=0.3); ax_scatter.set_ylabel(f"Recovered {label_prefix}")
    ax_scatter.tick_params(labelbottom=False)
    if show_legend: ax_scatter.legend(loc='lower right', fontsize=14, frameon=True)

    ax_resid.set_ylim(-15, 15); ax_resid.axhline(0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax_resid.grid(True, alpha=0.3); ax_resid.set_xlabel(f"True {label_prefix}"); ax_resid.set_ylabel("Residue")

def plot_six_panel_figure(df_all, df_supp, df_enh):
    fig = plt.figure(figsize=(18, 12))
    outer_grid = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.2, wspace=0.25)
    datasets = [("All Data", df_all), ("Suppressing", df_supp), ("Enhancing", df_enh)]

    for row_idx, (lab, is_pa) in enumerate([("V$_{rad}$ (km/s)", False), ("P.A. (deg)", True)]):
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
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==========================================
# 1. PATH CONFIG
# ==========================================
RING_MODEL = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/snr_res_testing/"
INITIAL_TRUE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/snr_res_testing/new_res_output"

# ==========================================
# 2. STYLE CONFIG (by SNR)
# ==========================================
SNR_STYLES = {
    5:  dict(color="#1f77b4", marker="o", label="SNR 5"),
    10: dict(color="#ff7f0e", marker="s", label="SNR 10"),
    20: dict(color="#2ca02c", marker="^", label="SNR 20"),
}

# ==========================================
# 3. FILENAME PARSER
# ==========================================
pat_fname = re.compile(
    r"^(?:\d+_)?ring_"
    r"v(?P<v>\d+)_R(?P<R>\d+)_"
    r"(?:(?:PAfixed)|PAk(?P<pa_k>-?\d*\.?\d+)_TP(?P<pa_tp>-?\d*\.?\d+))_"
    r"VR(?:(?:k(?P<vr_k>-?\d*\.?\d+)_TP(?P<vr_tp>-?\d*\.?\d+))|(?P<vr_fixed>0))_"
    r"INC(?P<inc>\d+)_Res(?P<res>\d+)\.txt$"
)

def compute_sign(pa_k, vr_k):
    if pa_k is None or vr_k is None or vr_k == 0:
        return 0
    return 1 if pa_k * vr_k > 0 else -1

# ==========================================
# 4. DATA LOADING
# ==========================================
def load_one_model_all_snr(fname):
    m = pat_fname.match(fname)
    if not m:
        return None

    g = m.groupdict()
    pa_k = float(g["pa_k"]) if g["pa_k"] else None
    vr_k = float(g["vr_k"]) if g["vr_k"] else None

    true_path = os.path.join(INITIAL_TRUE, fname)
    if not os.path.exists(true_path):
        return None

    df_true = pd.read_csv(true_path, delim_whitespace=True)
    true_vrad = df_true["VRAD(km/s)"].values
    true_pa   = df_true["P.A.(deg)"].values

    dfs = []

    for snr in [5, 10, 20]:
        folder = fname.replace(".txt", f"_SNR{snr}")
        out_path = os.path.join(
            RING_MODEL, folder, "second_step_check_sinw", "rings_final2.txt"
        )
        if not os.path.exists(out_path):
            continue

        df_out = pd.read_csv(out_path, delim_whitespace=True)
        n = min(len(true_vrad), len(df_out))

        df = pd.DataFrame({
            "True_VRAD": true_vrad[:n],
            "Measured_VRAD": df_out["VRAD(km/s)"][:n],
            "True_PA": true_pa[:n],
            "Measured_PA": df_out["P.A.(deg)"][:n],
            "SNR": snr,
            "pa_k": pa_k,
            "vr_k": vr_k,
            "sign": compute_sign(pa_k, vr_k),
        })

        df = df[df["True_VRAD"] != 0].reset_index(drop=True)
        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None

def load_all_data():
    dfs = []
    files = sorted(glob.glob(os.path.join(RING_MODEL, "*Res*.txt")))
    for fp in files:
        df = load_one_model_all_snr(os.path.basename(fp))
        if df is not None:
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# ==========================================
# 5. BIN STAT
# ==========================================
def bin_stats(x, y, nbins=10):
    df = pd.DataFrame({"x": x, "y": y})
    df["res"] = df["y"] - df["x"]
    bins = np.linspace(df["x"].min(), df["x"].max(), nbins + 1)
    df["bin"] = pd.cut(df["x"], bins=bins, include_lowest=True)
    g = df.groupby("bin", observed=False)
    xc = 0.5 * (bins[:-1] + bins[1:])
    return xc, g["res"].mean().values, g["res"].std().values

# ==========================================
# 6. SINGLE PANEL (same as original)
# ==========================================
def plot_combined_panel(ax_sc, ax_rs, x, y, snr_col, label, is_pa=False, show_legend=False):
    for snr, style in SNR_STYLES.items():
        idx = snr_col[snr_col == snr].index
        if len(idx) == 0:
            continue

        xs = x.loc[idx]
        ys = y.loc[idx]

        if is_pa:
            xs = xs - 120
            ys = ys - 120

        ax_sc.scatter(
            xs, ys, s=10, alpha=0.6,
            edgecolor="white", linewidth=0.3, **style
        )

        xc, mu, sig = bin_stats(xs, ys)
        ax_rs.plot(xc, mu, **style)
        ax_rs.fill_between(xc, mu - sig, mu + sig,
                           alpha=0.2, color=style["color"])

    ax_sc.plot([-30, 30], [-30, 30], "k--", lw=1, alpha=0.5)
    ax_sc.set_xlim(-30, 30)
    ax_sc.set_ylim(-30, 30)
    ax_sc.grid(alpha=0.3)
    ax_sc.set_ylabel(f"Measured {label}")
    ax_sc.tick_params(labelbottom=False)

    if show_legend:
        ax_sc.legend(loc="lower right", fontsize=12)
    ax_rs.set_ylim(-15, 15)
    ax_rs.axhline(0, ls="--", color="k", alpha=0.5)
    ax_rs.grid(alpha=0.3)
    ax_rs.set_xlabel(f"True {label}")
    ax_rs.set_ylabel("Residual")
    ax_sc.tick_params(axis="both", labelsize=13)
    ax_rs.tick_params(axis="both", labelsize=13)
# ==========================================
# 7. SIX-PANEL FIGURE (STRUCTURE IDENTICAL)
# ==========================================
def plot_six_panel_figure(df_all, df_supp, df_enh):
    fig = plt.figure(figsize=(18, 12))
    outer = gridspec.GridSpec(2, 3, hspace=0.25, wspace=0.25)

    datasets = [
        ("All Data", df_all),
        ("Suppressing", df_supp),
        ("Enhancing", df_enh),
    ]

    # --- Row 1: VRAD ---
    for i, (name, df) in enumerate(datasets):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, i],
            height_ratios=[3, 1], hspace=0.0
        )
        ax_sc = fig.add_subplot(inner[0])
        ax_rs = fig.add_subplot(inner[1], sharex=ax_sc)

        plot_combined_panel(
            ax_sc, ax_rs,
            df["True_VRAD"], df["Measured_VRAD"], df["SNR"],
            label="V$_{rad}$ (km/s)",
            is_pa=False,
            show_legend=(i == 0)
        )
        ax_sc.set_title(f"{name} - Vrad", fontsize=16, fontweight="bold")

    # --- Row 2: PA ---
    for i, (name, df) in enumerate(datasets):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1, i],
            height_ratios=[3, 1], hspace=0.0
        )
        ax_sc = fig.add_subplot(inner[0])
        ax_rs = fig.add_subplot(inner[1], sharex=ax_sc)

        plot_combined_panel(
            ax_sc, ax_rs,
            df["True_PA"], df["Measured_PA"], df["SNR"],
            label="P.A. (deg)",
            is_pa=True,
            show_legend=False
        )
        ax_sc.set_title(f"{name} - P.A.", fontsize=16, fontweight="bold")

    plt.savefig("figure_snr_six_panel.pdf")
    plt.show()

# ==========================================
# 8. MAIN
# ==========================================
if __name__ == "__main__":
    df_all = load_all_data()
    df_supp = df_all[df_all["sign"] > 0]
    df_enh  = df_all[df_all["sign"] < 0]

    plot_six_panel_figure(df_all, df_supp, df_enh)
