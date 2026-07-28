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
z_rs_to_cgh = 0.03          # RS平面からCGH平面までの距離 (30 mm)
k = 2 * np.pi / wavelength

# 画像パラメータ
I_views = 128                # 水平方向の視点数 (grid_x)
J_views = 128                # 垂直方向の視点数 (grid_y)
M_px = 32                    # 各投影画像の横解像度
N_px = 32                    # 各投影画像の縦解像度

# 全体のピクセル解像度 (4096 x 4096)
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> 全ピクセル数: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\Cone_multiview_output_fullparallax_128x128"
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*.png")))

if len(image_paths) == 0:
    raise FileNotFoundError("画像ファイルが見つかりません。パスを確認してください。")

# RS平面上の全波面配列の初期化 (複素数型)
u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

print("--> 視点画像のFFT処理およびRS平面波面の構成中...")

# 各視点画像 p_ij (M x N) を読み込み、2次元FFTを適用してRS平面に配置
for path in image_paths:
    filename = os.path.basename(path)
    parts = filename.replace(".png", "").split("_")
    # ファイル名規則: view_YY_XX.png または view_YYY_XXX.png
    iy, ix = int(parts[1]), int(parts[2])

    if iy >= J_views or ix >= I_views:
        continue

    # 画像読み込み (グレースケール: 0.0 〜 1.0)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # 振幅スペクトル（必要に応じてサイズ調整）
    if img.shape != (N_px, M_px):
        img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)

    # ランダム位相の生成 [0, 2π)
    random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))

    # 振幅（画像）にランダム位相を掛けて複素波面を作成
    complex_light = img * np.exp(1j * random_phase)

    # 複素波面に対して 2D-FFT を適用
    wavefront_block = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_light)))

    # RS平面の該当ブロック (iy, ix) に配置
    y_start = iy * N_px
    y_end = (iy + 1) * N_px
    x_start = ix * M_px
    x_end = (ix + 1) * M_px

    u_RS[y_start:y_end, x_start:x_end] = wavefront_block

print("--> RS平面波面の構成が完了しました。")

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

    # 伝達関数 H(fx, fy)
    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0
    H = np.exp(1j * k * z * np.sqrt(sq))

    # フーリエ領域での伝搬計算
    U_freq = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u_in)))
    U_prop_freq = U_freq * H
    u_out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(U_prop_freq)))

    return u_out

# =====================================================================
# 4. RS平面からCGH平面への光波伝搬 (z = 30 mm)
# =====================================================================
print(f"--> RS平面からCGH平面へ伝搬中 (距離 z = {z_rs_to_cgh*1e3:.1f} mm)...")
cgh_wavefront = propagate_asm(u_RS, z_rs_to_cgh, wavelength, pitch)

# =====================================================================
# 5. 指定角度からの像再生処理 (角スペクトル法: ASM + キャリア位相)
# =====================================================================
def propagate_asm_tilted(u_in, z, wavelength, pitch, theta_deg):
    Ny, Nx = u_in.shape
    theta_rad = np.radians(theta_deg)

    # 物理空間座標 [m]
    x = (np.arange(Nx) - Nx // 2) * pitch
    y = (np.arange(Ny) - Ny // 2) * pitch
    X, Y = np.meshgrid(x, y)

    # 観測角度 θ に応じたキャリア位相（斜め視点の再現）
    tilt_phase = np.exp(1j * k * np.sin(theta_rad) * X)
    u_tilted = u_in * tilt_phase

    # CGH平面から再生像平面（RS平面の位置 z）へ逆伝搬/伝搬
    u_out = propagate_asm(u_tilted, z, wavelength, pitch)

    return u_out

# =====================================================================
# 6. 各観測角度（-4°, -2°, 0°, 2°, 4°）での像再生と可視化
# =====================================================================
angles = [-4, -2, 0, 2, 4]
fig, axes = plt.subplots(1, len(angles), figsize=(20, 4.5))

extent_mm = [-N_x*pitch/2*1e3, N_x*pitch/2*1e3, -N_y*pitch/2*1e3, N_y*pitch/2*1e3]

print("--> 各観測角度からの像再生を計算中...")

for idx, angle in enumerate(angles):
    # CGH面からRS平面の位置 (-z_rs_to_cgh) へ戻して焦点を合わせる
    rs_wave = propagate_asm_tilted(cgh_wavefront, -z_rs_to_cgh, wavelength, pitch, angle)


    rec_intensity = np.abs(rs_wave) ** 2
    # 以下、正規化やプロットの処理が続く...

    # 正規化 (0.0 〜 1.0)
    max_val = np.max(rec_intensity)
    if max_val > 0:
        rec_intensity = rec_intensity / max_val

    im = axes[idx].imshow(rec_intensity, cmap='inferno', extent=extent_mm, vmin=0, vmax=1)
    axes[idx].set_title(f"RS View Angle: {angle}°")
    axes[idx].set_xlabel("x [mm]")
    if idx == 0:
        axes[idx].set_ylabel("y [mm]")

    cbar = fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    cbar.locator = ticker.MaxNLocator(nbins=5)
    cbar.formatter = ticker.FormatStrFormatter('%.2f')
    cbar.update_ticks()
    cbar.set_label("Normalized Intensity", fontsize=10)

plt.tight_layout()
plt.show()

# =====================================================================
# 7. CGH平面における「振幅ホログラム」と「位相ホログラム」の可視化
# =====================================================================
amplitude_map_cgh = np.abs(cgh_wavefront)
phase_map_cgh = np.angle(cgh_wavefront)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- 1. CGH 振幅ホログラム ---
im0 = axes[0].imshow(amplitude_map_cgh, cmap='gray', extent=extent_mm, aspect='equal')
axes[0].set_title("CGH Amplitude Map (at SLM Plane)", fontsize=13)
axes[0].set_xlabel("x [mm]", fontsize=11)
axes[0].set_ylabel("y [mm]", fontsize=11)

cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.locator = ticker.MaxNLocator(nbins=5)
cbar0.formatter = ticker.FormatStrFormatter('%.3f')
cbar0.update_ticks()
cbar0.set_label("Amplitude", fontsize=11)

# --- 2. CGH 位相ホログラム ---
im1 = axes[1].imshow(phase_map_cgh, cmap='twilight', extent=extent_mm, aspect='equal', vmin=-np.pi, vmax=np.pi)
axes[1].set_title("CGH Phase Map (at SLM Plane)", fontsize=13)
axes[1].set_xlabel("x [mm]", fontsize=11)
axes[1].set_ylabel("y [mm]", fontsize=11)

cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar1.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
cbar1.set_label("Phase [rad]", fontsize=11)

plt.tight_layout()
plt.show()