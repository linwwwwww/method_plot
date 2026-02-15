import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from astropy.visualization import PowerStretch, ImageNormalize

# --- 1. 全局样式设置 ---
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
mpl.rcParams['contour.negative_linestyle'] = 'solid' 
plt.rc('font', family='sans-serif', size=10)
params = {'text.usetex': False, 'mathtext.fontset': 'cm', 'mathtext.default': 'regular'}
plt.rcParams.update(params)

# --- 2. 路径与数据读取 ---
RING_BASE = "/net/dataserver3/data/users/linn/para_Barolo/test_more_model/"
RING_MODEL1123 = RING_BASE + "model1123/"
INITIAL_1123 = "/net/dataserver3/data/users/linn/para_initial/test_more_model/model1123/"
galaxy = "ring_v200_R50_PAk0.05_TP450_VRk0.09_TP300_INC60"
outfolder = RING_MODEL1123 + galaxy + "/second_step_check_sinw/"
gname = 'SNR10Res10mod'

recorver = pd.read_csv(outfolder + "rings_final2.txt", delimiter=r"\s+")
true_ = pd.read_csv(INITIAL_1123 + galaxy + ".txt", delimiter=r"\s+")
error_ = pd.read_csv(RING_MODEL1123 + galaxy + "/bootstrap_error.txt", delimiter=r"\s+")

# 读取文件用于 Top-Left
xmin, xmax, ymin, ymax = 238, 562, 238, 562
f1 = fits.open(outfolder + '/maps/' + gname + '_1mom.fits')
mom1 = f1[0].data[ymin:ymax+1, xmin:xmax+1]
mod_file_list = sorted([f for f in os.listdir(outfolder + '/maps/') if '1mom.fits' in f and ('azim' in f or 'local' in f)])
mom1_mod = fits.open(outfolder + '/maps/' + mod_file_list[0])[0].data[ymin:ymax+1, xmin:xmax+1]

rad, inc, pa, xpos, ypos, vsys = np.genfromtxt(
    outfolder + "rings_final2.txt",
    usecols=(1,4,5,9,10,11),
    unpack=True
)

xcen_m, ycen_m, inc_m, pa_m, vsys_m = np.nanmean((xpos, ypos, inc, pa, vsys), axis=1)
xcen = xcen_m - xmin
ycen = ycen_m - ymin

# --- 3. 开始绘图 ---
fig = plt.figure(figsize=(18, 12), dpi=150)
gs = GridSpec(2, 2, figure=fig,  wspace=0.2,height_ratios=[0.9, 1])

# ==========================================
# 【左上：速度图对比】
# ==========================================
# 将左上格子 (gs[0, 0]) 垂直切分：上方 85% 给图，下方 5% 给色标，中间留空
gs_tl_main = gs[0, 0].subgridspec(2, 1, height_ratios=[0.85, 0.1])
gs_tl_maps = gs_tl_main[0, 0].subgridspec(1, 2, wspace=0.05)

from astropy.visualization import PercentileInterval
from copy import copy

cmap_vel = plt.get_cmap('RdBu_r', 25)

# 用 DATA 来决定对称色标范围（推荐）
data_for_norm = mom1 - vsys_m

interval = PercentileInterval(99.5)
vmin, vmax = interval.get_limits(data_for_norm)
vabs = max(abs(vmin), abs(vmax))

norm_m = mpl.colors.Normalize(vmin=-vabs, vmax=vabs)

# 复制 cmap 并设置 NaN 为白色（和你大图一致）
cmap_vel = copy(cmap_vel)
cmap_vel.set_bad('w', 1.)

# 数据列表：[DATA, MODEL]
map_list = [mom1 - vsys_m, mom1_mod - vsys_m]
titles = ['DATA', 'MODEL']

for j, d in enumerate(map_list):
    ax = fig.add_subplot(gs_tl_maps[0, j])
    ax.imshow(d, origin='lower', cmap=cmap_vel, norm=norm_m, extent=[0, xmax-xmin, 0, ymax-ymin], aspect='equal')
    ax.plot(xcen, ycen, 'x', color='black', markersize=5)
    ax.contour(d, levels=[0], colors='green', origin='lower', linewidths=1)
    ax.set_xlabel('X (pix)')
    ax.set_title(titles[j], fontsize=10, fontweight='bold')
    
    if j == 0: 
        ax.set_ylabel('Y (pix)')
    else: 
        ax.set_yticklabels([])
    # ===== 加 global major axis =====
    x_grid = np.linspace(0, xmax-xmin, 500)
    
    # major axis: PA - 90°
    # y_major = np.tan(np.radians(pa_m - 90)) * (x_grid - xcen) + ycen
    # ax.plot(x_grid, y_major, '-', color='grey', lw=1.5, alpha=0.7)
    if len(rad_pix) > 5:
        x_p = rad_pix * np.cos(np.radians(pa - 90))
        y_p = rad_pix * np.sin(np.radians(pa - 90))
        ax.plot(xcen - x_p, ycen - y_p, '-', color='grey', lw=1.2, alpha=0.6)
        ax.plot(xcen + x_p, ycen + y_p, '-', color='grey', lw=1.2, alpha=0.6)
        

# 在左上格子的底部绘制横跨两图的色标
ax_cb_tl = fig.add_subplot(gs_tl_main[1, 0])
cb_tl = ColorbarBase(ax_cb_tl, orientation='horizontal', cmap=cmap_vel, norm=norm_m)
cb_tl.set_label(r'$\Delta V_{\mathrm{los}}$ (km s$^{-1}$)', fontsize=14)
cb_tl.outline.set_linewidth(0.5)

# 顶部标题
#fig.text(0.13, 0.9, r'$\Delta PA=7.5\ deg\ \ \Delta Vrad=-27\ km/s$', fontsize=14, fontweight='bold')

# ==========================================
# 【右上：PV 图对比】修正部分
# ==========================================
# 在右上格子里创建一个 1x2 的子布局
gs_tr = gs[0, 1].subgridspec(1, 2, wspace=0.05)

# PV 数据准备
zmin, zmax = 13, 187
image_maj = fits.open(outfolder + 'pvs/' + gname + '_pv_a.fits')
image_min = fits.open(outfolder + 'pvs/' + gname + '_pv_b.fits')
# 自动寻找对应的模型PV文件
mod_pv_a = sorted([f for f in os.listdir(outfolder + 'pvs/') if 'pv_a_azim.fits' in f or 'pv_a_local.fits' in f])[0]
mod_pv_b = sorted([f for f in os.listdir(outfolder + 'pvs/') if 'pv_b_azim.fits' in f or 'pv_b_local.fits' in f])[0]
im_mod_maj = fits.open(outfolder + 'pvs/' + mod_pv_a)
im_mod_min = fits.open(outfolder + 'pvs/' + mod_pv_b)

# 坐标计算 (简化用于展示)
head = image_maj[0].header
crpix, cdelt, crval = head['CRPIX1'], head['CDELT1'], head['CRVAL1']
xminpv, xmaxpv = int(crpix - 1 - 162), int(crpix - 1 + 162)
xmin_wcs = ((xminpv + 1 - 0.5 - crpix) * cdelt + crval) * 3600
xmax_wcs = ((xmaxpv + 1 + 0.5 - crpix) * cdelt + crval) * 3600
zmin_wcs, zmax_wcs = 255.25, -182.25
ext_pv = [xmin_wcs, xmax_wcs, zmin_wcs - vsys_m, zmax_wcs - vsys_m]

# 统一归一化
cont = 0.000675824
v_levels = np.array([1, 2, 4, 8, 16, 32, 64]) * cont
norm_pv = ImageNormalize(vmin=cont, vmax=image_maj[0].data.max()*0.7, stretch=PowerStretch(0.5))

# 绘制
pv_datas = [image_maj[0].data[zmin:zmax+1, xminpv:xmaxpv+1], image_min[0].data[zmin:zmax+1, xminpv:xmaxpv+1]]
pv_mods = [im_mod_maj[0].data[zmin:zmax+1, xminpv:xmaxpv+1], im_mod_min[0].data[zmin:zmax+1, xminpv:xmaxpv+1]]
phi_labels = [r'$\phi = 120^\circ$', r'$\phi = 210^\circ$']

for i in range(2):
    ax = fig.add_subplot(gs_tr[0, i])
    ax.imshow(pv_datas[i], origin='lower', cmap='Greys', norm=norm_pv, extent=ext_pv, aspect='auto', alpha=0.3)
    ax.contour(pv_datas[i], v_levels, origin='lower', linewidths=0.8, colors='#00008B', extent=ext_pv)
    ax.contour(pv_mods[i], v_levels, origin='lower', linewidths=1.2, colors='#B22222', extent=ext_pv)
    ax.axhline(y=0, color='black', lw=1); ax.axvline(x=0, color='black', lw=1)
    ax.set_xlabel('Offset (arcsec)')
    ax.text(0.8, 0.05, phi_labels[i], transform=ax.transAxes, fontsize=10, fontweight='bold', ha='center')
    if i == 0: 
            ax.set_ylabel(r'$\Delta V_{\mathrm{los}}$ (km s$^{-1}$)', fontsize=labsize)
            # 叠加旋转曲线点
            ax.plot(radius_pts, vlos_pts, 'y.', markersize=7, markeredgecolor='olive', alpha=0.6)
            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='#00008B', lw=1, label='DATA'),
                               Line2D([0], [0], color='#B22222', lw=1.5, label='MODEL')]
            ax.legend(handles=legend_elements, loc='upper left', frameon=True, fontsize=12)
    else:
            ax.set_yticklabels([]) # 隐藏右图 Y 轴标签
           
# ==========================================
# 【左下：VRAD 误差图】
# ==========================================
ax_vrad = fig.add_subplot(gs[1, 0])
ax_vrad.plot(true_["RAD(arcs)"], true_["VRAD(km/s)"], 'k--', label='True')
ax_vrad.errorbar(recorver["RAD(arcs)"], recorver["VRAD(km/s)"], yerr=error_["VRADerror(km/s)"], fmt='o', color='tab:blue', label='Recovered', markersize=4, capsize=2)
ax_vrad.set_xlabel('Radius (arcsec)')
ax_vrad.set_ylabel(r'$V_{\mathrm{rad}}$ (km/s)')
ax_vrad.legend(fontsize=14)
ax_vrad.set_ylim(-30, 30)
# ==========================================
# 【右下：PA 误差图】
# ==========================================
ax_pa = fig.add_subplot(gs[1, 1])
ax_pa.plot(true_["RAD(arcs)"], true_["P.A.(deg)"], 'k--', label='True')
ax_pa.errorbar(recorver["RAD(arcs)"], recorver["P.A.(deg)"], yerr=error_["P.A.error(deg)"], fmt='s', color='tab:red', label='Recovered', markersize=4, capsize=2)
ax_pa.set_xlabel('Radius (arcsec)')
ax_pa.set_ylabel('P.A. (deg)')
ax_pa.legend(fontsize=14)
ax_pa.set_ylim(120-10, 120+10)
plt.savefig("single_galaxy_error.pdf")
plt.tight_layout()
plt.show()