import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re # ★ 正規表現モジュールを追加

# =====================================================================
# 1. パラメータ設定 (論文仕様に基づく修正)
# =====================================================================
wavelength = 633e-9          # 光の波長 λ: 633 nm
pitch = 2.0e-6               # CGH平面のピクセルピッチ
k = 2 * np.pi / wavelength

# 論文に基づく距離パラメータ
D = 205e-3                    # 平面物体からCGHまでの距離 (205 mm)
z_rs = 5e-3                  # 平面物体からRS面までの距離 (5 mm)
z_rs_to_cgh = D - z_rs       # RS面からCGH面までの距離

# レンズ系パラメータ (結像シミュレーション用)
lens_distance = 200e-3       # CGHからレンズまでの距離 (200 mm)
pupil_diameter = 7e-3        # レンズの瞳直径 (7 mm)

# 画像パラメータ (論文に近い構成を想定)
I_views = 256                 # 水平方向の視点数
J_views = 256                 # 垂直方向の視点数
M_px = 32                    # 各投影画像の横解像度
N_px = 32                    # 各投影画像の縦解像度

# 全体のピクセル解像度
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> CGH Resolution: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\D_Diamond_multiview_output_fullparallax_256x256_z0010"

# フォルダ内のファイルを取得
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*")))

if len(image_paths) == 0:
    raise FileNotFoundError("画像ファイルが見つかりません。パスを確認してください。")

u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

for path in image_paths:
    filename = os.path.basename(path)
    
    # ★ 変更点: 正規表現で "view_YYY_XXX" の数字部分だけを確実に抽出する
    match = re.search(r"view_(\d+)_(\d+)", filename)
    if not match:
        # パターンに合致しないファイル（隠しファイルなど）はスキップ
        continue
        
    iy = int(match.group(1))
    ix = int(match.group(2))

    if iy >= J_views or ix >= I_views:
        continue

    # 画像読み込み
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
        
    img = img.astype(np.float32)

    if img.shape != (N_px, M_px):
        img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)

    img = np.sqrt(np.maximum(img, 0)) 

    # ランダム位相
    random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))
    complex_light = img * np.exp(1j * random_phase)

    # FFT処理
    RS_val = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_light)))
                             
    y_start = iy * N_px
    y_end = (iy + 1) * N_px
    x_start = ix * M_px
    x_end = (ix + 1) * M_px
    
    u_RS[y_start:y_end, x_start:x_end] = RS_val

print("--> RS plane is created.")

# =====================================================================
# 3. 帯域制限付き角スペクトル法 (Band-Limited ASM) 伝搬関数定義
# =====================================================================
def propagate_asm(u_in, z, wavelength, pitch):
    Ny, Nx = u_in.shape
    Lx = Nx * pitch
    Ly = Ny * pitch
    
    dfx = 1.0 / Lx
    dfy = 1.0 / Ly

    fx = (np.arange(Nx) - Nx // 2) * dfx
    fy = (np.arange(Ny) - Ny // 2) * dfy
    FX, FY = np.meshgrid(fx, fy)

    # 伝搬関数 H の計算
    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0  # エバネッセント波のカット
    H = np.exp(1j * k * z * np.sqrt(sq))

    # ★追加: バンドリミット (折り返しノイズ防止)
    # 伝搬距離 z と計算領域サイズ Lx, Ly に基づき、領域外へ飛び出す空間周波数を計算
    # f_max = (L/2) / (lambda * sqrt((L/2)^2 + z^2))
    limit_x = (Lx / 2) / np.sqrt((Lx / 2)**2 + z**2) / wavelength
    limit_y = (Ly / 2) / np.sqrt((Ly / 2)**2 + z**2) / wavelength

    # 領域外に到達する高周波数成分（広がりすぎる光）をカット
    H[(np.abs(FX) > limit_x) | (np.abs(FY) > limit_y)] = 0.0

    U_freq = np.fft.fftshift(np.fft.fft2(u_in))
    U_prop_freq = U_freq * H
    u_out = np.fft.ifft2(np.fft.ifftshift(U_prop_freq))
    
    return u_out

# =====================================================================
# 4. RS平面からCGH平面への光波伝搬 (物体光の計算)
# =====================================================================
print(f"--> RS plane to CGH plane ... (distance z = {z_rs_to_cgh*1e3:.1f} mm)...")
cgh_obj_complex = propagate_asm(u_RS, z_rs_to_cgh, wavelength, pitch)

# =====================================================================
# 5. 参照光の生成と干渉縞（振幅型CGH）の記録
# =====================================================================
x = (np.arange(N_x) - N_x // 2) * pitch
y = (np.arange(N_y) - N_y // 2) * pitch
X, Y = np.meshgrid(x, y)

theta_x = np.radians(1.5)
theta_y = np.radians(1.5)
ref_amp = np.max(np.abs(cgh_obj_complex))

# 参照波の生成
ref_wave = ref_amp * np.exp(1j * k * (X * np.sin(theta_x) + Y * np.sin(theta_y)))

# 干渉縞の強度計算 (正規化)
interference_intensity = np.abs(cgh_obj_complex + ref_wave)**2
cgh_amplitude = interference_intensity / np.max(interference_intensity)
print("--> Interference CGH is generated.")

# =====================================================================
# 6. レンズによる等倍結像シミュレーション (像再生)
# =====================================================================
print("--> Reconstructing image simulating lens imaging system...")

# 1. CGHに参照光を照射
cgh_illuminated = cgh_amplitude * ref_wave

# 2. CGHから「平面物体」の位置 (-D) へ逆伝搬
# （等倍結像の場合、物体面での波面が観測面にそのまま結像されます）
object_wave_rec = propagate_asm(cgh_illuminated, -D, wavelength, pitch)

# 3. レンズの瞳による解像度制限（空間周波数フィルタ）を適用
# 物体からレンズまでの総距離
L_total = D + lens_distance  
# レンズの開口数 (NA)
NA = (pupil_diameter / 2) / L_total  
# 遮断周波数 (カットオフ周波数)
cutoff_freq = NA / wavelength  

dfx = 1.0 / (N_x * pitch)
dfy = 1.0 / (N_y * pitch)
fx = (np.arange(N_x) - N_x // 2) * dfx
fy = (np.arange(N_y) - N_y // 2) * dfy
FX, FY = np.meshgrid(fx, fy)

# ローパスフィルタリング
pupil_filter = (FX**2 + FY**2) <= cutoff_freq**2
U_obj_freq = np.fft.fftshift(np.fft.fft2(object_wave_rec))
U_img_freq = U_obj_freq * pupil_filter
img_wave = np.fft.ifft2(np.fft.ifftshift(U_img_freq))

# 強度の計算
rec_intensity = np.abs(img_wave)**2
rec_intensity /= np.max(rec_intensity)

# =====================================================================
# 7. 描画 (振幅・位相・再生像)
# =====================================================================
# 物体光の複素振幅から位相（偏角）を抽出（位相型CGHデータ）
cgh_phase = np.angle(cgh_obj_complex)

extent_mm = [-N_x*pitch/2*1e3, N_x*pitch/2*1e3, -N_y*pitch/2*1e3, N_y*pitch/2*1e3]

# 1行3列のサブプロットを作成
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

# --- 1. 振幅型CGH（干渉縞）の表示 ---
im_amp = ax[0].imshow(cgh_amplitude, cmap='gray', extent=extent_mm, origin='lower')
ax[0].set_title("Amplitude CGH (Interference)")
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
cbar_amp = fig.colorbar(im_amp, ax=ax[0], fraction=0.046, pad=0.04)
cbar_amp.set_label("Transmittance", fontsize=10)

# --- 2. 位相型CGH（物体光の位相）の表示 ---
# 周期的な位相を表現するため、cmapには 'twilight' または 'hsv' が適しています
im_phase = ax[1].imshow(cgh_phase, cmap='twilight', extent=extent_mm, origin='lower', vmin=-np.pi, vmax=np.pi)
ax[1].set_title("Phase CGH (Object Phase)")
ax[1].set_xlabel("x [mm]")
ax[1].set_ylabel("y [mm]")
cbar_phase = fig.colorbar(im_phase, ax=ax[1], fraction=0.046, pad=0.04)
# カラーバーのメモリを π 表記にする
cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar_phase.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
cbar_phase.set_label("Phase [rad]", fontsize=10)

# --- 3. 再生像の表示 ---
im_rec = ax[2].imshow(rec_intensity, cmap='inferno', extent=extent_mm, origin='lower')
ax[2].set_title("Reconstructed Image at Imaging Plane")
ax[2].set_xlabel("x [mm]")
ax[2].set_ylabel("y [mm]")
cbar_rec = fig.colorbar(im_rec, ax=ax[2], fraction=0.046, pad=0.04)
cbar_rec.set_label("Intensity", fontsize=10)

plt.tight_layout()
plt.show()