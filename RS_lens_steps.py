import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

# =====================================================================
# 1. パラメータ設定 (本来の仕様に復元)
# =====================================================================
wavelength = 633e-9          # 光の波長 λ: 633 nm
pitch = 2.0e-6               # CGH平面のピクセルピッチ
k = 2 * np.pi / wavelength

D = 10e-3                    # 平面物体からCGHまでの距離
z_rs = 5e-3                  # 平面物体からRS面までの距離
z_rs_to_cgh = D - z_rs       # RS面からCGH面までの距離

lens_distance = 500e-3       # CGHからレンズまでの距離
pupil_diameter = 5e-3        # レンズの瞳直径

# 本番の視点数と解像度
I_views = 256 
J_views = 256 
M_px = 32 
N_px = 32 

N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> CGH Resolution: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

extent_mm = [-N_x*pitch/2*1e3, N_x*pitch/2*1e3, -N_y*pitch/2*1e3, N_y*pitch/2*1e3]

# =====================================================================
# 視覚化用の Figure 1 (生成フェーズ)
# =====================================================================
fig1 = plt.figure(figsize=(18, 10))
fig1.suptitle("Wavefront Conversion: Generation Phase", fontsize=16)

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算
# =====================================================================
# ご指定の実際のフォルダパス
image_folder = r"C:\Lab\D_Diamond_multiview_output_fullparallax_256x256_z0010"
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*")))

if len(image_paths) == 0:
    raise FileNotFoundError(f"画像が見つかりません。パスを確認してください: {image_folder}")

u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

# --- ★変更点: 代表視点として view_100_126 を検索して指定 ---
target_view = "view_100_126"
path = image_paths[0] # デフォルト値 (見つからなかった場合の保険)
for p in image_paths:
    if target_view in os.path.basename(p):
        path = p
        break

filename = os.path.basename(path)
match = re.search(r"view_(\d+)_(\d+)", filename)
if match:
    iy, ix = int(match.group(1)), int(match.group(2))
else:
    iy, ix = 0, 0

# --- Step 1: 代表視点画像の表示 ---
img_rep = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img_rep is None:
    raise ValueError(f"画像を読み込めませんでした: {path}")
img_rep = cv2.flip(img_rep, 0)  # ★この1行を追加（上下反転）
img_rep = img_rep.astype(np.float32)
if img_rep.shape != (N_px, M_px):
    img_rep = cv2.resize(img_rep, (M_px, N_px), interpolation=cv2.INTER_AREA)

ax1_1 = fig1.add_subplot(2, 3, 1)
ax1_1.imshow(img_rep, cmap='gray', origin='lower')
ax1_1.set_title(f"Step 1: Representative View Image\n({filename})")
ax1_1.axis('off')

# --- Step 2: ランダム位相の付加 ---
img_amp = np.sqrt(np.maximum(img_rep, 0))
random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))
complex_light = img_amp * np.exp(1j * random_phase)

ax1_2 = fig1.add_subplot(2, 3, 2)
im_phase1 = ax1_2.imshow(random_phase, cmap='twilight', origin='lower', vmin=0, vmax=2*np.pi)
ax1_2.set_title("Step 2: Random Phase\n(Added to Representative View)")
ax1_2.axis('off')
cbar1_2 = fig1.colorbar(im_phase1, ax=ax1_2, fraction=0.046, pad=0.04)
cbar1_2.set_ticks([0, np.pi, 2*np.pi])
cbar1_2.set_ticklabels(['0', 'π', '2π'])

# --- 残りの画像の処理とRS平面全体の作成 ---
print(f"--> Loading and processing {len(image_paths)} images...")
loaded_count = 0

for current_path in image_paths:
    current_filename = os.path.basename(current_path)
    # view_YYY_XXX にマッチ
    match = re.search(r"view_(\d+)_(\d+)", current_filename)
    if not match:
        continue
        
    iy_current = int(match.group(1))
    ix_current = int(match.group(2))
    
    if iy_current >= J_views or ix_current >= I_views:
        continue

    img_raw = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None: 
        continue
    img_raw = cv2.flip(img_raw, 0)  # ★この1行を追加（上下反転）
        
    img_raw = img_raw.astype(np.float32)
    if img_raw.shape != (N_px, M_px):
        img_raw = cv2.resize(img_raw, (M_px, N_px), interpolation=cv2.INTER_AREA)
        
    img_amp_local = np.sqrt(np.maximum(img_raw, 0)) 
    random_phase_local = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))
    complex_light_local = img_amp_local * np.exp(1j * random_phase_local)
    
    # FFT処理 (RS平面での波面)
    RS_val = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_light_local)))
                               
    y_start = iy_current * N_px
    y_end = (iy_current + 1) * N_px
    x_start = ix_current * M_px
    x_end = (ix_current + 1) * M_px
    
    u_RS[y_start:y_end, x_start:x_end] = RS_val
    
    loaded_count += 1
    if loaded_count % 5000 == 0:
        print(f"    Processed {loaded_count} / {len(image_paths)} images...")

print("--> RS plane is successfully created.")

# --- Step 3: 全体のRS平面波面の強度 ---
ax1_3 = fig1.add_subplot(2, 3, 3)
im_rs = ax1_3.imshow(np.log10(np.abs(u_RS)**2 + 1e-9), cmap='inferno', extent=extent_mm, origin='lower')
ax1_3.set_title("Step 3: RS Plane Wavefront Intensity\n(Log10 Scale)")
ax1_3.set_xlabel("x [mm]")
ax1_3.set_ylabel("y [mm]")
fig1.colorbar(im_rs, ax=ax1_3, fraction=0.046, pad=0.04)

# =====================================================================
# 3. 帯域制限付き角スペクトル法 (Band-Limited ASM) 伝搬関数定義
# =====================================================================
def propagate_asm(u_in, z, wavelength, pitch):
    Ny, Nx = u_in.shape
    Lx, Ly = Nx * pitch, Ny * pitch
    dfx, dfy = 1.0 / Lx, 1.0 / Ly
    
    # メモリ節約のためfloat32でメッシュグリッドを生成
    fx = ((np.arange(Nx) - Nx // 2) * dfx).astype(np.float32)
    fy = ((np.arange(Ny) - Ny // 2) * dfy).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)
    
    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0
    H = np.exp(1j * k * z * np.sqrt(sq))
    
    limit_x = (Lx / 2) / np.sqrt((Lx / 2)**2 + z**2) / wavelength
    limit_y = (Ly / 2) / np.sqrt((Ly / 2)**2 + z**2) / wavelength
    
    H[(np.abs(FX) > limit_x) | (np.abs(FY) > limit_y)] = 0.0
    
    U_freq = np.fft.fftshift(np.fft.fft2(u_in))
    U_prop_freq = U_freq * H
    u_out = np.fft.ifft2(np.fft.ifftshift(U_prop_freq))
    return u_out

# =====================================================================
# 4. RS平面からCGH平面への光波伝搬 (物体光の計算)
# =====================================================================
print(f"--> Propagating RS plane to CGH plane (distance z = {z_rs_to_cgh*1e3:.1f} mm)...")
cgh_obj_complex = propagate_asm(u_RS, z_rs_to_cgh, wavelength, pitch)

# --- Step 4: CGH平面での物体光の振幅と位相 ---
ax1_4 = fig1.add_subplot(2, 3, 4)
# データが大きいのでダウンサンプリングして表示（メモリ/表示負荷軽減）
skip = 4 
im_obj_amp = ax1_4.imshow(np.abs(cgh_obj_complex[::skip, ::skip]), cmap='gray', extent=extent_mm, origin='lower')
ax1_4.set_title("Step 4a: Object Wavefront Amplitude\nat CGH Plane")
ax1_4.set_xlabel("x [mm]")
ax1_4.set_ylabel("y [mm]")
fig1.colorbar(im_obj_amp, ax=ax1_4, fraction=0.046, pad=0.04)

ax1_5 = fig1.add_subplot(2, 3, 5)
im_obj_phase = ax1_5.imshow(np.angle(cgh_obj_complex[::skip, ::skip]), cmap='twilight', extent=extent_mm, origin='lower', vmin=-np.pi, vmax=np.pi)
ax1_5.set_title("Step 4b: Object Wavefront Phase\nat CGH Plane")
ax1_5.set_xlabel("x [mm]")
ax1_5.set_ylabel("y [mm]")
cbar1_5 = fig1.colorbar(im_obj_phase, ax=ax1_5, fraction=0.046, pad=0.04)
cbar1_5.set_ticks([-np.pi, 0, np.pi])
cbar1_5.set_ticklabels(['-π', '0', 'π'])

# =====================================================================
# 5. 参照光の生成と干渉縞（振幅型CGH）の記録
# =====================================================================
print("--> Generating interference CGH...")
x = ((np.arange(N_x) - N_x // 2) * pitch).astype(np.float32)
y = ((np.arange(N_y) - N_y // 2) * pitch).astype(np.float32)
X, Y = np.meshgrid(x, y)
theta_x, theta_y = np.radians(1.5), np.radians(1.5)
ref_amp = np.max(np.abs(cgh_obj_complex))

# 参照波の生成
ref_wave = ref_amp * np.exp(1j * k * (X * np.sin(theta_x) + Y * np.sin(theta_y)))

# 干渉縞の強度計算 (正規化)
interference_intensity = np.abs(cgh_obj_complex + ref_wave)**2
cgh_amplitude = interference_intensity / np.max(interference_intensity)

# メモリ解放
del interference_intensity 

print("--> Interference CGH is generated.")

# --- Step 6: 振幅型CGH (干渉縞) ---
ax1_6 = fig1.add_subplot(2, 3, 6)
im_cgh_amp = ax1_6.imshow(cgh_amplitude[::skip, ::skip], cmap='gray', extent=extent_mm, origin='lower')
ax1_6.set_title("Step 6: Amplitude CGH\n(Interference Fringe Intensity)")
ax1_6.set_xlabel("x [mm]")
ax1_6.set_ylabel("y [mm]")
fig1.colorbar(im_cgh_amp, ax=ax1_6, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# =====================================================================
# 視覚化用の Figure 2 (再生フェーズ)
# =====================================================================
fig2 = plt.figure(figsize=(18, 10))
fig2.suptitle("Wavefront Conversion: Reconstruction Phase", fontsize=16)

# =====================================================================
# 6. レンズによる等倍結像シミュレーション (像再生)
# =====================================================================
print("--> Reconstructing image simulating lens imaging system...")

# 1. CGHに参照光を照射
cgh_illuminated = cgh_amplitude * ref_wave
del cgh_amplitude, ref_wave  # メモリ解放

# 2. CGHから「平面物体」の位置 (-D) へ逆伝搬
object_wave_rec = propagate_asm(cgh_illuminated, -D, wavelength, pitch)
del cgh_illuminated

# 3. レンズの瞳による解像度制限（空間周波数フィルタ）を適用
L_total = D + lens_distance 
NA = (pupil_diameter / 2) / L_total 
cutoff_freq = NA / wavelength 

dfx, dfy = 1.0 / (N_x * pitch), 1.0 / (N_y * pitch)
fx = ((np.arange(N_x) - N_x // 2) * dfx).astype(np.float32)
fy = ((np.arange(N_y) - N_y // 2) * dfy).astype(np.float32)
FX, FY = np.meshgrid(fx, fy)

# --- Step 8: レンズ瞳フィルタの視覚化 ---
pupil_filter = (FX**2 + FY**2) <= cutoff_freq**2

ax2_2 = fig2.add_subplot(2, 2, 2)
im_pupil = ax2_2.imshow(pupil_filter[::skip, ::skip], cmap='gray', extent=[fx[0]/1e3, fx[-1]/1e3, fy[0]/1e3, fy[-1]/1e3], origin='lower')
ax2_2.set_title("Step 8a: Lens Pupil Filter\n(Binary Map in Frequency Space)")
ax2_2.set_xlabel("fx [lines/mm]")
ax2_2.set_ylabel("fy [lines/mm]")
fig2.colorbar(im_pupil, ax=ax2_2, fraction=0.046, pad=0.04)

# ローパスフィルタリング
U_obj_freq = np.fft.fftshift(np.fft.fft2(object_wave_rec))
U_img_freq = U_obj_freq * pupil_filter

# フィルタ適用後の空間周波数強度 (視覚化)
ax2_3 = fig2.add_subplot(2, 2, 3)
im_filtered_freq = ax2_3.imshow(np.log10(np.abs(U_img_freq[::skip, ::skip])**2 + 1e-9), cmap='inferno', extent=[fx[0]/1e3, fx[-1]/1e3, fy[0]/1e3, fy[-1]/1e3], origin='lower')
ax2_3.set_title("Step 8b: Filtered Spatial Frequency Intensity\n(Log10 Scale)")
ax2_3.set_xlabel("fx [lines/mm]")
ax2_3.set_ylabel("fy [lines/mm]")
fig2.colorbar(im_filtered_freq, ax=ax2_3, fraction=0.046, pad=0.04)

img_wave = np.fft.ifft2(np.fft.ifftshift(U_img_freq))
del U_obj_freq, U_img_freq

# 最終的な再生像の強度の計算
rec_intensity = np.abs(img_wave)**2
rec_intensity /= np.max(rec_intensity)

# --- Step 7: 物体平面での再生像 (フィルタ適用前) ---
ax2_1 = fig2.add_subplot(2, 2, 1)
im_obj_rec = ax2_1.imshow(np.abs(object_wave_rec[::skip, ::skip])**2, cmap='inferno', extent=extent_mm, origin='lower')
ax2_1.set_title("Step 7: Back-propagated Wavefront\nat Object Plane (Before Filtering)")
ax2_1.set_xlabel("x [mm]")
ax2_1.set_ylabel("y [mm]")
fig2.colorbar(im_obj_rec, ax=ax2_1, fraction=0.046, pad=0.04)

# --- Step 9: 最終再生像 (Imaging Plane) ---
ax2_4 = fig2.add_subplot(2, 2, 4)
im_final = ax2_4.imshow(rec_intensity[::skip, ::skip], cmap='inferno', extent=extent_mm, origin='lower')
ax2_4.set_title("Step 9: Final Reconstructed Image\nat Imaging Plane")
ax2_4.set_xlabel("x [mm]")
ax2_4.set_ylabel("y [mm]")
fig2.colorbar(im_final, ax=ax2_4, fraction=0.046, pad=0.04)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

print("--> Done! Displaying figures...")
plt.show()