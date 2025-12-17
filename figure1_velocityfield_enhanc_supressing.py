import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase
from astropy.io import fits
from astropy.visualization import PercentileInterval, PowerStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from copy import copy
import warnings
import glob

# ================= 基础设置 =================
warnings.filterwarnings("ignore")

mpl.rc('xtick', direction='in', top=True)
mpl.rc('ytick', direction='in', right=True)
mpl.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('font', family='sans-serif', serif='Helvetica', size=10)
params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'}
plt.rcParams.update(params)

# ★★★ 请根据服务器修改此处 ★★★
base_path = '/net/'  

# ================= 模型路径配置 =================
models_raw = [
    ("Pure", "dataserver3/data/users/linn/para_Barolo/test_more_model/ring_v200_R50_PAfixed_VR0_INC60/second_step_check_sinw/"),
    ("Vrad", "dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/ring_v200_R50_PAfixed_VRk-0.09_TP300_INC60/second_step_check_sinw/"),
    ("PA (+)", "dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/ring_v200_R50_PAk-0.1_TP300_VR0_INC60/second_step_check_sinw/"),
    ("Suppressing", "dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/ring_v200_R50_PAk-0.1_TP300_VRk-0.09_TP300_INC60/second_step_check_sinw/"),
    ("PA (-)", "dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/ring_v200_R50_PAk0.1_TP300_VR0_INC60/second_step_check_sinw/SNR10Res10mod_maps_azim.pdf"), 
    ("Enhancing", "dataserver3/data/users/linn/para_Barolo/test_more_model/model1123/ring_v200_R50_PAk0.1_TP300_VRk-0.09_TP300_INC60/second_step_check_sinw/")
]

# 1. 清洗路径 (处理 pdf 结尾, 拼接 base_path)
models_clean = []
for title, path in models_raw:
    if path.endswith('.pdf'):
        path = os.path.dirname(path) + '/'
    if not path.endswith('/'):
        path += '/'
    full_path = os.path.join(base_path, path)
    models_clean.append((title, full_path))

# 保持顺序
models_ordered = models_clean

# ================= 全局参数 =================
gname = 'SNR10Res10mod'
xmin, xmax = 250, 550
ymin, ymax = 250, 550
zmin, zmax = 10, 190
twostage = 1 
cmap_vel = plt.get_cmap('RdBu_r', 25)

# ================= 辅助函数定义 =================

def get_rings_data(outfolder):
    """读取 rings_final2.txt 获取详细几何参数(PA array, Rad array等)用于画线"""
    fname = 'rings_final2.txt' if twostage else 'rings_final1.txt'
    path = os.path.join(outfolder, fname)
    try:
        # 读取列: 1:Rad, 4:Inc, 5:PA, 9:Xpos, 10:Ypos, 11:Vsys
        data = np.genfromtxt(path, usecols=(1, 4, 5, 9, 10, 11), unpack=True)
        rad, inc, pa, xpos, ypos, vsys = data
        
        return {
            'rad': rad, 'pa': pa, 
            'vsys_m': np.nanmean(vsys),
            'xcen_m': np.nanmean(xpos),
            'ycen_m': np.nanmean(ypos),
            'pa_m': np.nanmean(pa),
            'inc_m': np.nanmean(inc)
        }
    except Exception as e:
        print(f"Error reading rings from {path}: {e}")
        return None

def plot_mom1(ax, outfolder, p_data, is_bottom_row):
    """
    第1列：Data Velocity Field
    包含：Global Major(黑虚), Warped Axis(灰实/点), Green Contour(V=0)
    """
    if p_data is None: 
        ax.text(0.5,0.5, "No Rings Data", ha='center')
        return None, None
    
    # 解包数据
    vsys_m = p_data['vsys_m']
    xcen = p_data['xcen_m'] - xmin
    ycen = p_data['ycen_m'] - ymin
    rad = p_data['rad']
    pa = p_data['pa']
    pa_m = p_data['pa_m']
    
    # 读取 Data Map (注意：这里读取的是 DATA 的 mom1)
    fpath = os.path.join(outfolder, 'maps', gname + '_azim_1mom.fits')
    f1 = fits.open(fpath)
    mom1 = f1[0].data[-10+ymin:ymax+10, -10+xmin:xmax+10]
    f1.close()
    
    data_to_plot = mom1 - vsys_m
    ext = [0, xmax - xmin, 0, ymax - ymin]

    # Normalize
    interval = PercentileInterval(99.5)
    vmin, vmax = interval.get_limits(data_to_plot)
    vabs = max(abs(vmin), abs(vmax))
    norm = mpl.colors.Normalize(vmin=-vabs, vmax=vabs)
    
    cmap = copy(cmap_vel)
    cmap.set_bad('w', 1.)
    
    # 绘图
    im = ax.imshow(data_to_plot, origin='lower', cmap=cmap, norm=norm, aspect='equal', extent=ext, interpolation='nearest')
    ax.plot(xcen, ycen, 'x', color='#000000', markersize=7, mew=1.5)

    # --- 绘制辅助线 (基于您的代码逻辑) ---
    x_grid = np.arange(0, xmax - xmin, 0.1)
    
    # 1. Global Major Axis (Black Dashed)
    y_maj = np.tan(np.radians(pa_m - 90)) * (x_grid - xcen) + ycen
    ax.plot(x_grid, y_maj, '--', color='k', linewidth=1, alpha=0.8)

    # 2. Global Minor Axis (Grey Solid)
    y_min = np.tan(np.radians(pa_m)) * (x_grid - xcen) + ycen
    ax.plot(x_grid, y_min, '--', color='k', linewidth=1, alpha=0.8)
    ax.set_xlim(-10, xmax - xmin+10)
    ax.set_ylim(-10, ymax - ymin+10)
    
    # 去除 ticks 和 labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    if is_bottom_row:
        pass 
        
    

    # 3. Warped Axis / Locus (复用您的代码)
    rad_pix = rad / 4.0  # 您的缩放比例
    try: nr = len(rad_pix)
    except: nr = 1
    
    # 绘制弯曲的 Major Axis
    if nr < 10:
        x_pix = rad_pix * np.cos(np.radians(pa_m - 90))
        y_pix = rad_pix * np.sin(np.radians(pa_m - 90))
        ax.scatter(x_pix + xcen, y_pix + ycen, c='grey', s=12)
        ax.scatter(xcen - x_pix, ycen - y_pix, c='grey', s=12)
    
    if nr > 5 and not np.all(np.diff(pa) == 0):
        x_pix = rad_pix * np.cos(np.radians(pa - 90))
        y_pix = rad_pix * np.sin(np.radians(pa - 90))
        ax.plot(xcen - x_pix, ycen - y_pix, '-', color='grey', lw=1)
        ax.plot(x_pix + xcen, y_pix + ycen, '-', color='grey', lw=1)

    # 4. Isovelocity Contour (Green, Level=0)
    maskmap = np.copy(mom1)
    maskmap[mom1==mom1] = 1
    ax.contour(data_to_plot * maskmap, levels=[0], colors='green', origin='lower', extent=ext, linewidths=1.2)
    
    # 坐标轴清理
    # 限制显示范围为 Data 切片大小，防止线条画出界
    
    if not os.path.exists(fpath):
        # 尝试读取 local 或 azim 作为替代，如果 data 不存在
        fpath = os.path.join(outfolder, 'maps', gname + '_azim_1mom.fits')
        if not os.path.exists(fpath):
             ax.text(0.5, 0.5, "Data Map Missing", ha='center')
             return None, None

    return im, norm

def plot_pv_hybrid(ax, outfolder, pv_type, p_data, is_left_col, is_bottom):
    """
    第2、3列：PV Diagrams
    背景：Data (Grey)
    Contour：Model (Red only)
    """
    vsys_m = p_data['vsys_m']
    suffix_data = '_pv_a.fits' if pv_type == 'major' else '_pv_b.fits'
    
    # 读取 Data PV
    f_data_path = os.path.join(outfolder, 'pvs', gname + suffix_data)
    if not os.path.exists(f_data_path): 
        ax.text(0.5, 0.5, "Data PV Missing", ha='center')
        return

    f = fits.open(f_data_path)
    head = f[0].header
    
    # WCS 计算
    crpix = head['CRPIX1']
    cdelt = head['CDELT1']
    crval = head['CRVAL1']
    naxis1 = head['NAXIS1']
    xminpv = max(0, np.floor(crpix - 1 - 150))
    xmaxpv = min(naxis1 - 1, np.ceil(crpix - 1 + 150))
    xmin_wcs = ((xminpv + 1 - 0.5 - crpix) * cdelt + crval) * 3600
    xmax_wcs = ((xmaxpv + 1 + 0.5 - crpix) * cdelt + crval) * 3600
    zmin_wcs, zmax_wcs = 262.75, -189.75 # Hardcoded
    
    ext = [xmin_wcs, xmax_wcs, zmin_wcs - vsys_m, zmax_wcs - vsys_m]
    data_crop = f[0].data[zmin:zmax+1, int(xminpv):int(xmaxpv)+1]
    f.close()
    
    # Normalize Data (Greyscale)
    cont = 0.000598286
    interval = PercentileInterval(99.5)
    vmax_norm = interval.get_limits(data_crop)[1]
    if np.isnan(vmax_norm): vmax_norm = cont*10
    norm = ImageNormalize(vmin=cont, vmax=vmax_norm, stretch=PowerStretch(0.5))
    
    ax.imshow(data_crop, origin='lower', cmap=mpl.cm.Greys, norm=norm, extent=ext, aspect='auto')
    
    # 读取并叠加 Model PV (Red Contours)
    suffix_mod = f'pv_{"a" if pv_type=="major" else "b"}_azim.fits'
    # 查找 azim 或 local
    mod_files = glob.glob(os.path.join(outfolder, 'pvs', '*' + suffix_mod))
    if not mod_files:
        suffix_mod = f'pv_{"a" if pv_type=="major" else "b"}_local.fits'
        mod_files = glob.glob(os.path.join(outfolder, 'pvs', '*' + suffix_mod))

    if mod_files:
        f_mod = fits.open(mod_files[0])
        mod_crop = f_mod[0].data[zmin:zmax+1, int(xminpv):int(xmaxpv)+1]
        v_levels = np.array([1, 2, 4, 8, 16, 32, 64]) * cont
        # 只画红色
        ax.contour(mod_crop, v_levels, origin='lower', linewidths=1, colors='#B22222', extent=ext)
        f_mod.close()
    
    # 辅助线
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Labels
#     if is_left_col:
#         ax.set_ylabel(r'$\Delta V_{LOS}$ (km/s)')
#     else:
#         ax.set_yticklabels([])
        
#     if is_bottom:
#         ax.set_xlabel('Offset (arcsec)')
#     else:
#         ax.set_xticklabels([])
def get_initial_profile_path(barolo_path):
    """
    将 Barolo 输出路径转换为 Initial Parameter 的 txt 路径
    例如: 
    From: .../para_Barolo/.../ring_v200...INC60/second_step_check_sinw/
    To:   .../para_initial/.../ring_v200...INC60.txt
    """
    # 1. 规范化路径，去除末尾可能存在的 '/'
    p = os.path.normpath(barolo_path)
    
    # 2. 如果路径以 .pdf 结尾，去掉文件名回到目录
    if p.endswith('.pdf'):
        p = os.path.dirname(p)
        
    # 3. 如果末尾是 'second_step_check_sinw'，向上退一级
    if os.path.basename(p) == 'second_step_check_sinw':
        p = os.path.dirname(p)
        
    # 4. 此时 p 应该以 model 文件夹结尾 (e.g., ring_v200...INC60)
    # 替换路径中的 'para_Barolo' 为 'para_initial'
    new_path_base = p.replace('para_Barolo', 'para_initial')
    
    # 5. 加上 .txt 后缀
    txt_path = new_path_base + ".txt"
    
    return txt_path

def plot_real_profile(ax, barolo_path, row_idx, is_bottom_row):
    """
    读取对应的 txt 文件并绘制 PA 和 VRAD 在同一个 Y 轴上
    """
    # 获取转换后的 txt 路径
    txt_path = get_initial_profile_path(barolo_path)
    
    # 检查文件是否存在
    if not os.path.exists(txt_path):
        ax.text(0.5, 0.5, "Txt Not Found", ha='center', fontsize=8)
        print(f"Missing: {txt_path}")
        return

    try:
        # --- 数据读取部分 (保持不变) ---
        try:
            data = np.genfromtxt(txt_path, names=True, encoding=None)
            colnames = [n.upper() for n in data.dtype.names]
            col_rad = next(n for n in colnames if 'RAD' in n)
            col_pa  = next(n for n in colnames if 'P.A' in n or 'PA' in n)
            col_vrad= next(n for n in colnames if 'VRAD' in n)
            
            rad  = data[data.dtype.names[colnames.index(col_rad)]]
            pa   = data[data.dtype.names[colnames.index(col_pa)]]
            vrad = data[data.dtype.names[colnames.index(col_vrad)]]
        except:
            data_arr = np.genfromtxt(txt_path)
            rad  = data_arr[:, 0]
            pa   = data_arr[:, 1]
            vrad = data_arr[:, 2]

        # === 开始绘图 ===
        color_pa = '#2ca02c'   # Green
        color_vr = '#9467bd'   # Purple
        
        # 数据处理
        pa = pa - 120 

        # 1. 绘制 P.A. (实线) -> 使用 ax
        ax.plot(rad, pa, '-', color=color_pa, lw=2, label='$\Delta$ P.A. (deg)')
        
        # 2. 绘制 VRAD (虚线) -> 同样使用 ax (去掉 twinx)
        ax.plot(rad, vrad, '--', color=color_vr, lw=2, label='$\Delta$ Vrad (km/s)')
        
        # 设置 X 轴范围
        ax.set_xlim(left=0, right=np.max(rad)*1.05)
        
        # 设置 Y 轴范围 (自动适应两条线的数据)
        # 获取两组数据的最大最小值来设定范围，或者让 matplotlib 自动处理
        all_y = np.concatenate([pa, vrad])
        if len(all_y) > 0:
            y_min, y_max = np.min(all_y), np.max(all_y)
            margin = (y_max - y_min) * 0.1
            ax.set_ylim(-32, 32)

        ax.grid(True, ls=':', alpha=0.5)

        # Labels
        if is_bottom_row:
            ax.set_xlabel('Radius (arcsec)')
        else:
            ax.tick_params(labelbottom=False)

        # === 关键修改：单轴图例 ===
        if row_idx == 0:
            # 既然共用一个轴，就不写具体的单位 Label 了，或者写个通用的
            # ax.set_ylabel('Value') 
            
            # 直接显示图例，包含单位
            ax.legend(loc='upper right', fontsize=8, frameon=False)
        else:
            # 其他行如果不需要显示Y轴刻度数字，可以去掉
            # ax.set_yticklabels([]) 
            pass

    except Exception as e:
        ax.text(0.5, 0.5, "Read Error", ha='center', fontsize=8)
        print(f"Error reading {txt_path}: {e}")
# ================= 主程序执行 =================

fig = plt.figure(figsize=(15, 18))
# 4列: Map | PV Major | PV Minor | Profile
gs = gridspec.GridSpec(6, 4, width_ratios=[1, 1.2, 1.2, 1], wspace=0.15, hspace=0.08)
rows = 6

global_im, global_norm = None, None

for i in range(rows):
    title, path = models_ordered[i]
    is_bottom = (i == rows - 1)
    print(f"Plotting Row {i+1}: {title}...")
    
    # 获取详细数据用于画线
    p_data = get_rings_data(path)
    if p_data is None: 
        print(f"Skipping row {i} (no data)")
        continue
    
    # --- 1. Data Velocity Field ---
    ax0 = fig.add_subplot(gs[i, 0])
    # 传入 p_data 以绘制 Major/Minor 轴
    im, norm = plot_mom1(ax0, path, p_data, is_bottom)
    if i == 0: 
        global_im, global_norm = im, norm
        ax0.set_title("Data Velocity Field")
    
    # 行标题 (左侧)
    ax0.text(-0.15, 0.5, title, transform=ax0.transAxes, 
             va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)

    # --- 2. PV Major (Data=Grey, Model=Red) ---
    ax1 = fig.add_subplot(gs[i, 1])
    plot_pv_hybrid(ax1, path, 'major', p_data, is_left_col=True, is_bottom=is_bottom)
    if i == 0: ax1.set_title("PV Major")

    # --- 3. PV Minor (Data=Grey, Model=Red) ---
    ax2 = fig.add_subplot(gs[i, 2])
    plot_pv_hybrid(ax2, path, 'minor', p_data, is_left_col=False, is_bottom=is_bottom)
    if i == 0: ax2.set_title("PV Minor")
    
    # --- 4. Profile (Placeholder) ---
    ax3 = fig.add_subplot(gs[i, 3])
    plot_real_profile(ax3, path, i, is_bottom)
    if i == 0: ax3.set_title("Profile")
# --- 添加 Colorbar (Velocity Field) ---
# if global_im:
#     # 获取最后一行的位置
#     pos_last = fig.add_subplot(gs[rows-1, 0]).get_position()
#     # 在下方添加 colorbar
#     cax = fig.add_axes([pos_last.x0, pos_last.y0 - 0.04, pos_last.width, 0.01])
#     cb = ColorbarBase(cax, orientation='horizontal', cmap=cmap_vel, norm=global_norm)
#     cb.set_label(r'$\Delta V_{LOS}$ (km/s)', fontsize=9)

# 保存文件
output_file = 'Final_Kinematic_Comparison_6x4.pdf'
plt.subplots_adjust(left=0.1, bottom=0.08, right=0.95, top=0.95)
fig.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Finished. Saved to {output_file}")