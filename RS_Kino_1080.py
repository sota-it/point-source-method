import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.fft as sfft  

# =====================================================================
# 1. パラメータ設定
# =====================================================================
wavelength = 532e-9          # 光の波長 λ: 532 nm
pitch = 8.0e-6               # CGH平面の初期ピクセルピッチ (高解像度計算用)
k = 2 * np.pi / wavelength

# 距離パラメータ
D = 105e-3                   # 平面物体からCGHまでの距離 (105 mm)
z_rs = 5e-3                  # 平面物体からRS面までの距離 (5 mm)
z_rs_to_cgh = D - z_rs       # RS面からCGH面までの距離

# レンズ系パラメータ (結像シミュレーション用)
lens_distance = 200e-3       # CGHからレンズまでの距離 (200 mm)
pupil_diameter = 7e-3        # レンズの瞳直径

# 画像パラメータ
I_views = 128                
J_views = 128                
M_px = 8                    
N_px = 8                    

# 元のピクセル解像度 (1024 x 1024)
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> Original RS Resolution: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群 (Light Field) の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\8px_letterP_multiview_output_fullparallax_128x128_z0005"

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
# 2.5 RS平面のゼロパディング (エイリアシングノイズ防止)
# =====================================================================
print("--> Applying Zero-Padding to RS plane (2N x 2N)...")
N_x_pad = 2 * N_x
N_y_pad = 2 * N_y

u_RS_padded = np.zeros((N_y_pad, N_x_pad), dtype=np.complex64)

y_offset = N_y // 2
x_offset = N_x // 2
u_RS_padded[y_offset : y_offset + N_y, x_offset : x_offset + N_x] = u_RS

del u_RS 

# =====================================================================
# 3. 帯域制限付き角スペクトル法 (Band-Limited ASM) 伝搬関数定義
# =====================================================================
def propagate_asm(u_in, z, wavelength, current_pitch):
    Ny, Nx = u_in.shape
    Lx, Ly = Nx * current_pitch, Ny * current_pitch
    
    dfx, dfy = 1.0 / Lx, 1.0 / Ly

    fx = ((np.arange(Nx) - Nx // 2) * dfx).astype(np.float32)
    fy = ((np.arange(Ny) - Ny // 2) * dfy).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)

    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0
    
    H = np.exp(1j * k * z * np.sqrt(sq)).astype(np.complex64)

    limit_x = (Lx / 2) / np.sqrt((Lx / 2)**2 + z**2) / wavelength
    limit_y = (Ly / 2) / np.sqrt((Ly / 2)**2 + z**2) / wavelength

    H[(np.abs(FX) > limit_x) | (np.abs(FY) > limit_y)] = 0.0

    del FX, FY, sq
    
    U_freq = np.fft.fftshift(sfft.fft2(u_in, workers=-1))
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

# =====================================================================
# 5. 実機SLMへの適合とバイナリホログラムの生成
# =====================================================================
slm_size = 1024  
slm_pitch = 8.0e-6  

print(f"--> Extracting {slm_size}x{slm_size} and generating Binary CGH...")

cy, cx = N_y_pad // 2, N_x_pad // 2
cgh_original_area = cgh_obj_complex[cy - N_y//2 : cy + N_y//2, cx - N_x//2 : cx + N_x//2]
del cgh_obj_complex 

real_part = np.real(cgh_original_area).astype(np.float32)
imag_part = np.imag(cgh_original_area).astype(np.float32)
del cgh_original_area

real_resized = cv2.resize(real_part, (slm_size, slm_size), interpolation=cv2.INTER_AREA)
imag_resized = cv2.resize(imag_part, (slm_size, slm_size), interpolation=cv2.INTER_AREA)
del real_part, imag_part

cgh_obj_slm = real_resized + 1j * imag_resized
del real_resized, imag_resized


x_slm = (np.arange(slm_size) - slm_size // 2) * slm_pitch
y_slm = (np.arange(slm_size) - slm_size // 2) * slm_pitch
X_slm, Y_slm = np.meshgrid(x_slm, y_slm)

phase_raw = np.angle(cgh_obj_slm)

# cos(phase)の符号を判定し、位相を 0 または π に二値化する
cgh_phase_binary = np.where(np.cos(phase_raw) >= 0, 0.0, np.pi)

# 8bit量子化 (0=0, π=128 にマッピング)
slm_8bit = np.round((cgh_phase_binary / (2 * np.pi)) * 255).astype(np.uint8)

del cgh_obj_slm, phase_raw, X_slm, Y_slm
print(f"--> Binary Phase CGH generated. Size: {slm_8bit.shape}")

# =====================================================================
# 6. レンズによる等倍結像シミュレーション (像再生)
# =====================================================================
print("--> Reconstructing image simulating lens imaging system...")

rec_size = 1024

slm_phase_reconstructed = (slm_8bit.astype(np.float32) / 255.0) * 2 * np.pi
cgh_illuminated = np.zeros((rec_size, rec_size), dtype=np.complex64)

c_y, c_x = rec_size // 2, rec_size // 2
cgh_illuminated[c_y - slm_size//2 : c_y + slm_size//2, c_x - slm_size//2 : c_x + slm_size//2] = np.exp(1j * slm_phase_reconstructed)

object_wave_rec = propagate_asm(cgh_illuminated, -D, wavelength, slm_pitch)
del cgh_illuminated

L_total = D + lens_distance  
NA = (pupil_diameter / 2) / L_total  
cutoff_freq = NA / wavelength  

dfx, dfy = 1.0 / (rec_size * slm_pitch), 1.0 / (rec_size * slm_pitch)
fx = ((np.arange(rec_size) - rec_size // 2) * dfx).astype(np.float32)
fy = ((np.arange(rec_size) - rec_size // 2) * dfy).astype(np.float32)
FX, FY = np.meshgrid(fx, fy)

pupil_filter = (FX**2 + FY**2) <= cutoff_freq**2
del FX, FY

U_obj_freq = np.fft.fftshift(sfft.fft2(object_wave_rec, workers=-1))
U_obj_freq *= pupil_filter
del pupil_filter

img_wave = sfft.ifft2(np.fft.ifftshift(U_obj_freq), workers=-1)
del U_obj_freq, object_wave_rec

raw_rec_intensity = np.abs(img_wave)**2
plot_raw_rec_intensity = raw_rec_intensity.copy()
plot_raw_rec_intensity /= np.max(plot_raw_rec_intensity)

rec_intensity = raw_rec_intensity ** 0.5 
vmax_val = np.percentile(rec_intensity, 99.9)
rec_intensity = np.clip(rec_intensity / vmax_val, 0.0, 1.0)
plot_rec_intensity = rec_intensity.copy()
del img_wave, raw_rec_intensity, rec_intensity

# =====================================================================
# 7. 描画 (振幅・位相・再生像)
# =====================================================================
print("--> Generating figures...")
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

slm_extent_mm = [-slm_size*slm_pitch/2*1e3, slm_size*slm_pitch/2*1e3, -slm_size*slm_pitch/2*1e3, slm_size*slm_pitch/2*1e3]
rec_extent_mm = [-rec_size*slm_pitch/2*1e3, rec_size*slm_pitch/2*1e3, -rec_size*slm_pitch/2*1e3, rec_size*slm_pitch/2*1e3]

# --- 1. 二値化位相CGHの表示 ---
# 位相 0 と π をはっきり見せるための設定
im_phase = ax[0].imshow(cgh_phase_binary, cmap='gray', extent=slm_extent_mm, origin='lower', vmin=0, vmax=np.pi)
ax[0].set_title("Binary Phase CGH\n(0 and π)")
ax[0].set_xlabel("x [mm]")
ax[0].set_ylabel("y [mm]")
cbar_phase = fig.colorbar(im_phase, ax=ax[0], fraction=0.046, pad=0.04)
cbar_phase.set_ticks([0, np.pi])
cbar_phase.set_ticklabels(['0', 'π'])
cbar_phase.set_label("Phase [rad]", fontsize=10)

# --- 2. 生の再生強度 ---
im_rec_raw = ax[1].imshow(plot_raw_rec_intensity, cmap='inferno', extent=rec_extent_mm, origin='lower', vmin=0, vmax=1.0)
ax[1].set_title("Reconstructed Image\n(Raw Intensity, Normalized)")
ax[1].set_xlabel("x [mm]")
ax[1].set_ylabel("y [mm]")
cbar_rec_raw = fig.colorbar(im_rec_raw, ax=ax[1], fraction=0.046, pad=0.04)
cbar_rec_raw.set_label("Normalized Intensity", fontsize=10)

# --- 3. 再生像の全体表示 (エンハンス済) ---
im_rec_full = ax[2].imshow(plot_rec_intensity, cmap='inferno', extent=rec_extent_mm, origin='lower', vmin=0, vmax=1.0)
ax[2].set_title("Reconstructed Image\n(Full Field, Enhanced)")
ax[2].set_xlabel("x [mm]")
ax[2].set_ylabel("y [mm]")
cbar_rec_full = fig.colorbar(im_rec_full, ax=ax[2], fraction=0.046, pad=0.04)
cbar_rec_full.set_label("Intensity", fontsize=10)

plt.tight_layout()
plt.show()