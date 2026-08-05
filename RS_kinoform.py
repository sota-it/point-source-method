import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.fft as sfft  # ★メモリ節約＆高速化のためにSciPyのFFTを使用

# =====================================================================
# 1. パラメータ設定
# =====================================================================
wavelength = 532e-9          # 光の波長 λ: 532 nm
pitch = 2.0e-6               # CGH平面のピクセルピッチ
k = 2 * np.pi / wavelength

# 距離パラメータ
D = 105e-3                   # 平面物体からCGHまでの距離 (105 mm)
z_rs = 5e-3                  # 平面物体からRS面までの距離 (5 mm)
z_rs_to_cgh = D - z_rs       # RS面からCGH面までの距離

# レンズ系パラメータ (結像シミュレーション用)
lens_distance = 200e-3       # CGHからレンズまでの距離 (200 mm)
pupil_diameter = 7e-3       # レンズの瞳直径

# 画像パラメータ
I_views = 256                # 水平方向の視点数
J_views = 256                # 垂直方向の視点数
M_px = 65                    # 各投影画像の横解像度 (★奇数に変更)
N_px = 65                    # 各投影画像の縦解像度 (★奇数に変更)

# 元のピクセル解像度
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> Original RS Resolution: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\2nd_65px_Diamond_multiview_output_fullparallax_256x256_z0010"
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*")))

if len(image_paths) == 0:
    raise FileNotFoundError("画像ファイルが見つかりません。パスを確認してください。")

u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

for path in image_paths:
    filename = os.path.basename(path)
    match = re.search(r"view_(\d+)_(\d+)", filename)
    if not match:
        continue
        
    iy = int(match.group(1))
    ix = int(match.group(2))


    if iy >= J_views or ix >= I_views:
        continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
        
    # 上下反転（座標系の補正）
    img = cv2.flip(img, 0)
    img = img.astype(np.float32)

    if img.shape != (N_px, M_px):
        img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)

    img = np.sqrt(np.maximum(img, 0)) 

    random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))
    complex_light = img * np.exp(1j * random_phase)

    RS_val = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_light)))
                             
    y_start, y_end = iy * N_px, (iy + 1) * N_px
    x_start, x_end = ix * M_px, (ix + 1) * M_px
    
    u_RS[y_start:y_end, x_start:x_end] = RS_val

print("--> RS plane is created.")

# =====================================================================
# ★2.5 RS平面のゼロパディング (エイリアシングノイズ防止)
# =====================================================================
print("--> Applying Zero-Padding to RS plane (2N x 2N)...")
N_x_pad = 2 * N_x
N_y_pad = 2 * N_y

u_RS_padded = np.zeros((N_y_pad, N_x_pad), dtype=np.complex64)

# 画像の中央に元の波面を配置
y_offset = N_y // 2
x_offset = N_x // 2
u_RS_padded[y_offset : y_offset + N_y, x_offset : x_offset + N_x] = u_RS

del u_RS # 古い配列を削除してメモリ解放

print(f"--> Padded CGH Resolution: {N_x_pad} x {N_y_pad}")
extent_mm_pad = [-N_x_pad*pitch/2*1e3, N_x_pad*pitch/2*1e3, -N_y_pad*pitch/2*1e3, N_y_pad*pitch/2*1e3]

# =====================================================================
# 3. 帯域制限付き角スペクトル法 (Band-Limited ASM) 伝搬関数定義
# =====================================================================
def propagate_asm(u_in, z, wavelength, pitch):
    Ny, Nx = u_in.shape
    Lx, Ly = Nx * pitch, Ny * pitch
    
    dfx, dfy = 1.0 / Lx, 1.0 / Ly

    # メモリ節約のため float32 で計算
    fx = ((np.arange(Nx) - Nx // 2) * dfx).astype(np.float32)
    fy = ((np.arange(Ny) - Ny // 2) * dfy).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)

    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0
    
    # ★ 1jを使うとcomplex128になるため、明示的に complex64 にキャスト
    H = np.exp(1j * k * z * np.sqrt(sq)).astype(np.complex64)

    limit_x = (Lx / 2) / np.sqrt((Lx / 2)**2 + z**2) / wavelength
    limit_y = (Ly / 2) / np.sqrt((Ly / 2)**2 + z**2) / wavelength

    H[(np.abs(FX) > limit_x) | (np.abs(FY) > limit_y)] = 0.0

    # 重いメッシュグリッドを即座にメモリ解放
    del FX, FY, sq
    
    # ★ SciPyを用いて並列処理(workers=-1)で高速化
    U_freq = np.fft.fftshift(sfft.fft2(u_in, workers=-1))
    
    # 新しい配列を作らず、既存の U_freq に掛け算して上書き (メモリ節約)
    U_freq *= H
    del H
    
    u_out = sfft.ifft2(np.fft.ifftshift(U_freq), workers=-1)
    
    return u_out

# =====================================================================
# 4. RS平面からCGH平面への光波伝搬 (物体光の計算)
# =====================================================================
print(f"--> Propagating RS plane to CGH plane (distance z = {z_rs_to_cgh*1e3:.1f} mm)...")
cgh_obj_complex = propagate_asm(u_RS_padded, z_rs_to_cgh, wavelength, pitch)
del u_RS_padded

# プロット用に間引いた位相データを保存 (描画時のメモリクラッシュ防止)
skip = 8
plot_obj_phase = np.angle(cgh_obj_complex[::skip, ::skip])

# =====================================================================
# 5. キノフォーム型位相ホログラムの記録 (論文の式8に準拠)
# =====================================================================
print("--> Generating Kinoform Phase CGH...")

# ★ 式(8): H = arg(W_H) に従い、複素振幅から位相（偏角）を直接抽出するだけ
cgh_phase = np.angle(cgh_obj_complex).astype(np.float32)

plot_cgh_phase = cgh_phase[::skip, ::skip].copy()
del cgh_obj_complex

print("--> Kinoform Phase CGH is generated.")

# =====================================================================
# 6. レンズによる等倍結像シミュレーション (像再生)
# =====================================================================
print("--> Reconstructing image simulating lens imaging system...")

# 1. キノフォーム型位相CGHに真っ直ぐな平面波（振幅1、位相0）を照射して透過
# ※参照光（斜めの波）は掛けません
cgh_illuminated = np.exp(1j * cgh_phase).astype(np.complex64)
del cgh_phase

# 2. CGHから「平面物体」の位置 (-D) へ逆伝搬
object_wave_rec = propagate_asm(cgh_illuminated, -D, wavelength, pitch)
del cgh_illuminated

# --- (これ以降のレンズ瞳フィルタ適用等のコードは元のまま変更なし) ---

# 3. レンズの瞳による解像度制限（空間周波数フィルタ）を適用
L_total = D + lens_distance  
NA = (pupil_diameter / 2) / L_total  
cutoff_freq = NA / wavelength  

dfx, dfy = 1.0 / (N_x_pad * pitch), 1.0 / (N_y_pad * pitch)
fx = ((np.arange(N_x_pad) - N_x_pad // 2) * dfx).astype(np.float32)
fy = ((np.arange(N_y_pad) - N_y_pad // 2) * dfy).astype(np.float32)
FX, FY = np.meshgrid(fx, fy)

# ローパスフィルタリング
pupil_filter = (FX**2 + FY**2) <= cutoff_freq**2
del FX, FY

# ★ SciPy FFTを使用
U_obj_freq = np.fft.fftshift(sfft.fft2(object_wave_rec, workers=-1))
# インプレースでフィルタリング
U_obj_freq *= pupil_filter
del pupil_filter

# IFFT
img_wave = sfft.ifft2(np.fft.ifftshift(U_obj_freq), workers=-1)
del U_obj_freq, object_wave_rec

# 強度の計算
rec_intensity = np.abs(img_wave)**2
rec_intensity /= np.max(rec_intensity)
plot_rec_intensity = rec_intensity[::skip, ::skip].copy()
del img_wave, rec_intensity

# =====================================================================
# 7. 描画 (振幅・位相・再生像)
# =====================================================================
print("--> Generating figures...")
# 3つの画像を並べるため、サブプロットを 1行3列 に変更し横幅を広げる
fig, ax = plt.subplots(1, 3, figsize=(22, 6))

# --- 1. 位相型CGH（キノフォーム）の表示 ---
im_phase = ax[0].imshow(plot_cgh_phase, cmap='twilight', extent=extent_mm_pad, origin='lower', vmin=-np.pi, vmax=np.pi)
ax[0].set_title("Phase CGH (Kinoform)")
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
cbar_phase = fig.colorbar(im_phase, ax=ax[0], fraction=0.046, pad=0.04)
cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar_phase.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
cbar_phase.set_label("Phase [rad]", fontsize=10)

# --- 2. 再生像の表示 (ズームなし・全体) ---
im_rec_full = ax[1].imshow(plot_rec_intensity, cmap='inferno', extent=extent_mm_pad, origin='lower')
ax[1].set_title("Reconstructed Image (Full Field)")
ax[1].set_xlabel("x [mm]")
ax[1].set_ylabel("y [mm]")
cbar_rec_full = fig.colorbar(im_rec_full, ax=ax[1], fraction=0.046, pad=0.04)
cbar_rec_full.set_label("Intensity", fontsize=10)

# --- 3. 再生像の表示 (中央ズーム) ---
im_rec_zoom = ax[2].imshow(plot_rec_intensity, cmap='inferno', extent=extent_mm_pad, origin='lower')
ax[2].set_title("Reconstructed Image (Zoomed)")
ax[2].set_xlabel("x [mm]")
ax[2].set_ylabel("y [mm]")

# 中央の指定した範囲（ミリメートル）だけをズームアップして表示
zoom_range = 16.64  # 表示したい範囲（±16.64 mm の場合）
ax[2].set_xlim(-zoom_range, zoom_range)
ax[2].set_ylim(-zoom_range, zoom_range)

cbar_rec_zoom = fig.colorbar(im_rec_zoom, ax=ax[2], fraction=0.046, pad=0.04)
cbar_rec_zoom.set_label("Intensity", fontsize=10)

plt.tight_layout()
plt.show()