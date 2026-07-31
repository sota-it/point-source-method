import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =====================================================================
# 1. パラメータ設定 (論文 Opt. Express 19, 9086 (2011) 表2に準拠)
# =====================================================================
wavelength = 633e-9          # 光の波長 λ: 633 nm
pitch = 2.0e-6               # RS平面およびCGH平面のピクセルピッチ (2 μm)
z_rs_to_cgh = 0.5           # RS平面からCGH平面までの距離 (30 mm)
k = 2 * np.pi / wavelength

# 画像パラメータ
I_views = 256                # 水平方向の視点数 (grid_x)
J_views = 256                # 垂直方向の視点数 (grid_y)
M_px = 32                    # 各投影画像の横解像度
N_px = 32                    # 各投影画像の縦解像度

# 全体のピクセル解像度 (4096 x 4096)
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> RS_px: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\Sphere_multiview_output_fullparallax_256x256_z0010"
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*.png")))

if len(image_paths) == 0:
    raise FileNotFoundError("画像ファイルが見つかりません。パスを確認してください。")

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算 (修正版)
# =====================================================================
u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

for path in image_paths:
    filename = os.path.basename(path)
    parts = filename.replace(".png", "").split("_")
    iy, ix = int(parts[1]), int(parts[2])

    if iy >= J_views or ix >= I_views:
        continue

    # 画像読み込み
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if img.shape != (N_px, M_px):
        img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)

    img = np.sqrt(np.maximum(img, 0))  # 強度 -> 振幅

    # ランダム位相
    random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))

    complex_light = img * np.exp(1j * random_phase)

    RS_val = np.fft.fft2(complex_light)

    y_start = iy * N_px
    y_end = (iy + 1) * N_px
    x_start = ix * M_px
    x_end = (ix + 1) * M_px

    u_RS[y_start:y_end, x_start:x_end] = RS_val

print("--> RS plane is created.")

# =====================================================================
# 3. 角スペクトル法 (ASM) 伝搬関数定義
# =====================================================================
def propagate_asm(u_in, z, wavelength, pitch):
    """角スペクトル法 (Angular Spectrum Method) による自由空間光伝搬"""
    Ny, Nx = u_in.shape

    dfx = 1.0 / (Nx * pitch)
    dfy = 1.0 / (Ny * pitch)

    fx = (np.arange(Nx) - Nx // 2) * dfx
    fy = (np.arange(Ny) - Ny // 2) * dfy
    FX, FY = np.meshgrid(fx, fy)

    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0
    H = np.exp(1j * k * z * np.sqrt(sq))

    U_freq = np.fft.fft2(u_in)
    U_prop_freq = U_freq * H
    u_out = np.fft.ifft2(U_prop_freq)

    return u_out

# =====================================================================
# 4. RS平面からCGH平面への光波伝搬 (角スペクトル法: ASM)
# =====================================================================
print(f"--> RS plane to CGH plane ... (distance z = {z_rs_to_cgh*1e3:.1f} mm)...")
cgh_complex = propagate_asm(u_RS, z_rs_to_cgh, wavelength, pitch)

# =====================================================================
# 4.5 位相型データ (Phase-Only CGH) の抽出
# =====================================================================
# 実部と虚部から偏角 (位相 rad) を計算
cgh_phase = np.angle(cgh_complex)

# 振幅を 1 に固定し、位相データのみを持つ波面を生成 (位相型SLM表示用)
cgh_phase_only_wavefront = np.exp(1j * cgh_phase)

print("--> Phase-Only CGH is generated.")

# =====================================================================
# 6. 位相型データからの像再生（観測角度 0°）
# =====================================================================
extent_mm = [-N_x*pitch/2*1e3, N_x*pitch/2*1e3, -N_y*pitch/2*1e3, N_y*pitch/2*1e3]

print(f"--> Phase-Only CGH to RS plane ... (distance z = {z_rs_to_cgh*1e3:.1f} mm)...")

# 位相データのみの波面を CGH平面から RS平面の位置 (-z) へ逆伝搬
rs_wave_rec = propagate_asm(cgh_phase_only_wavefront, -z_rs_to_cgh, wavelength, pitch)
rec_intensity = np.abs(rs_wave_rec) ** 2

max_intensity = np.max(rec_intensity)
rec_intensity /= max_intensity  # 正規化 (0~1)

# 像再生の可視化
fig, ax = plt.subplots(figsize=(6, 5.5))
im = ax.imshow(rec_intensity, cmap='gray', extent=extent_mm, origin='lower')

ax.set_title("Reconstructed View from Phase-Only CGH (ASM, 0°)")
ax.set_xlabel("x [mm]")
ax.set_ylabel("y [mm]")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Intensity [a.u.]", fontsize=10)

plt.tight_layout()
plt.show()

# =====================================================================
# 7. CGH平面における位相ホログラムの可視化
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 6))

im_phase = ax.imshow(cgh_phase, cmap='twilight', extent=extent_mm, aspect='equal', vmin=-np.pi, vmax=np.pi, origin='lower')
ax.set_title("Extracted Phase-Only CGH Data (at SLM Plane)", fontsize=12)
ax.set_xlabel("x [mm]", fontsize=10)
ax.set_ylabel("y [mm]", fontsize=10)

cbar = fig.colorbar(im_phase, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
cbar.set_label("Phase [rad]", fontsize=10)

plt.tight_layout()
plt.show()