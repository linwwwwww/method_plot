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
    """
    计算 sign:
    sign > 0 (同号) -> 定义为 Suppressing
    sign < 0 (异号) -> 定义为 Enhancing
    """
    if pa_k is None: return 0
    if vr_k is None: return 0
    if vr_k == 0: return 0
    return 1 if pa_k * vr_k > 0 else -1

# ==========================================
# 3. 数据加载函数 (DATA LOADING)
# ==========================================
def load_single_file(file_name, ring_root):
    """读取单个文件的 True vs Measured 数据"""
    m = pat_fname.match(file_name)
    if not m: return None

    g = m.groupdict()
    pa_k = float(g["pa_k"]) if g.get("pa_k") else None
    vr_k = float(g["vr_k"]) if g.get("vr_k") else None
    inc_val = int(g["inc"])

    # 1. 确定 Initial (True) 路径
    if "model1123" in ring_root:
        in_path = os.path.join(INITIAL_1123, file_name)
    else:
        in_path = os.path.join(INITIAL_BASE, file_name)

    if not os.path.exists(in_path): return None

    try: inputf = pd.read_csv(in_path, delimiter=r"\s+")
    except: return None

    # 2. 确定 Measured (Output) 路径
    base_name, _ = os.path.splitext(file_name)
    out_path = os.path.join(ring_root, base_name, "second_step_check_sinw", "rings_final2.txt")
    if not os.path.exists(out_path): return None

    try: outputf = pd.read_csv(out_path, delimiter=r"\s+")
    except: return None

    # 3. 对齐与合并
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
    
    # 过滤无效点
    df = df[df["True_VRAD"] != 0].reset_index(drop=True)
    return df

def load_all_data():
    """读取所有符合条件的数据"""
    df_list = []
    
    # 1. model1123: INC45, INC75, INC60
    files_1123 = sorted(glob.glob(os.path.join(RING_MODEL1123, "*.txt")))
    for fp in files_1123:
        fname = os.path.basename(fp)
        if "v200" not in fname: continue
        if "INC45" in fname or "INC75" in fname or "INC60" in fname:
            df = load_single_file(fname, RING_MODEL1123)
            if df is not None: df_list.append(df)

    # 2. test_more_model (Base): INC60
    files_base = sorted(glob.glob(os.path.join(RING_BASE, "*.txt")))
    for fp in files_base:
        fname = os.path.basename(fp)
        if "v200" not in fname: continue
        if "INC60" in fname:
            df = load_single_file(fname, RING_BASE)
            if df is not None: df_list.append(df)

    if not df_list:
        print("No data loaded!")
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)

# ==========================================
# 4. 绘图与统计函数 (PLOTTING)
# ==========================================
def bin_stats(x_plot, y_plot, nbins=10):
    """计算残差的 Mean 和 1-Sigma"""
    temp_df = pd.DataFrame({'x': np.array(x_plot), 'y': np.array(y_plot)})
    temp_df['res'] = temp_df['y'] - temp_df['x']

    if temp_df['x'].empty:
        return [], [], []
        
    bin_edges = np.linspace(temp_df['x'].min(), temp_df['x'].max(), nbins + 1)
    temp_df['bin_cat'] = pd.cut(temp_df['x'], bins=bin_edges, include_lowest=True)

    grouped = temp_df.groupby('bin_cat', observed=False)
    mean_res = grouped['res'].mean().values
    sigma_res = grouped['res'].std().values
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, mean_res, sigma_res

def plot_combined_panel(ax_scatter, ax_resid, x, y, inc_col, label_prefix, is_pa=False, show_legend=False):
    """
    画单个Panel: 上部 Scatter, 下部 Residual
    """
    if x.empty or y.empty: return

    for inc, style in INC_STYLES.items():
        sub = inc_col[inc_col == inc].index
        if len(sub) == 0: continue
        
        x_sub = x.loc[sub]
        y_sub = y.loc[sub]
        
        if is_pa:
            x_plot = x_sub - 120
            y_plot = y_sub - 120
        else:
            x_plot = x_sub
            y_plot = y_sub
            
        # --- Scatter ---
        ax_scatter.scatter(
            x_plot, y_plot,
            s=10, alpha=0.6, edgecolor="white", linewidth=0.3,
            **style
        )
        
        # --- Residual ---
        bin_centers, mean_res, sigma_res = bin_stats(x_plot, y_plot)
        if len(bin_centers) > 0:
            ax_resid.plot(bin_centers, mean_res, **style)
            fill_style = {k: v for k, v in style.items() if k not in ['marker', 'label']}
            ax_resid.fill_between(bin_centers, mean_res - sigma_res, mean_res + sigma_res, alpha=0.2, **fill_style)

    # --- 装饰 Scatter ---
    ax_scatter.set_xlim(-30, 30)
    ax_scatter.set_ylim(-30, 30)
    ax_scatter.plot([-30, 30], [-30, 30], "k--", lw=1, alpha=0.5)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.set_ylabel(f"Measured {label_prefix}")
    ax_scatter.tick_params(labelbottom=False) # 隐藏Scatter的x轴标签
    
    # 图例只在需要的 Panel 显示
    if show_legend:
        ax_scatter.legend(loc='lower right', fontsize=14, frameon=True)

    # --- 装饰 Residue ---
    ax_resid.set_ylim(-15, 15)
    ax_resid.axhline(0, color='k', linestyle='--', lw=1, alpha=0.5)
    ax_resid.grid(True, alpha=0.3)
    ax_resid.set_xlabel(f"True {label_prefix}")
    ax_resid.set_ylabel("Residue")
    # sharex 已经自动处理了 X 轴范围

def plot_six_panel_figure(df_all, df_supp, df_enh):
    """
    绘制 2排3列 的大图
    Row 1: VRAD (All, Suppressing, Enhancing)
    Row 2: PA   (All, Suppressing, Enhancing)
    每个子图内部: 上2/3 Scatter, 下1/3 Residual
    """
    fig = plt.figure(figsize=(18, 12))
    
    # 使用 GridSpec 定义外层布局: 2行, 3列
    # height_ratios=[1, 1] 表示上下两排高度一致
    outer_grid = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.2, wspace=0.25)

    # 数据集列表 [ (Title, DataFrame), ... ]
    datasets = [
        ("All Data", df_all),
        ("Suppressing", df_supp),
        ("Enhancing", df_enh)
    ]

    # ==========================
    # 第一排: VRAD
    # ==========================
    for col_idx, (name, df) in enumerate(datasets):
        # 在外层 Grid 的对应位置创建一个内层 GridSpec (2行1列, 3:1)
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, 
            subplot_spec=outer_grid[0, col_idx], 
            height_ratios=[3, 1], 
            hspace=0.0  # Scatter 和 Residue 紧贴
        )
        
        ax_sc = fig.add_subplot(inner_grid[0])
        ax_rs = fig.add_subplot(inner_grid[1], sharex=ax_sc)
        
        # 仅在第一列显示图例
        show_leg = (col_idx == 0)
        
        plot_combined_panel(
            ax_sc, ax_rs, 
            df["True_VRAD"], df["Measured_VRAD"], df["INC"], 
            label_prefix="V$_{rad}$ (km/s)", is_pa=False, show_legend=show_leg
        )
        
        ax_sc.set_title(f"{name} - Vrad", fontsize=16, fontweight='bold')

    # ==========================
    # 第二排: P.A.
    # ==========================
    for col_idx, (name, df) in enumerate(datasets):
        # 在外层 Grid 的对应位置创建一个内层 GridSpec
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, 
            subplot_spec=outer_grid[1, col_idx], 
            height_ratios=[3, 1], 
            hspace=0.0
        )
        
        ax_sc = fig.add_subplot(inner_grid[0])
        ax_rs = fig.add_subplot(inner_grid[1], sharex=ax_sc)
        
        plot_combined_panel(
            ax_sc, ax_rs, 
            df["True_PA"], df["Measured_PA"], df["INC"], 
            label_prefix="P.A. (deg)", is_pa=True, show_legend=False
        )
        
        ax_sc.set_title(f"{name} - P.A.", fontsize=16, fontweight='bold')
    plt.savefig("figure3-incforincfixedgalaxies.pdf")
    plt.show()

# ==========================================
# 5. 主执行逻辑 (MAIN)  
# ==========================================
if __name__ == "__main__":
    print("Loading all data...")
    df_all = load_all_data()
    print(f"Total loaded: {len(df_all)} points.")
    
    if not df_all.empty:
        # 拆分数据集
        df_supp = df_all[df_all["sign"] > 0].reset_index(drop=True)
        df_enh  = df_all[df_all["sign"] < 0].reset_index(drop=True)
        
        print(f"Suppressing count: {len(df_supp)}")
        print(f"Enhancing count:   {len(df_enh)}")

        # 绘图
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
