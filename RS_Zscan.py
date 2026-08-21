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
pitch = 8.0e-6               # CGH平面の初期ピクセルピッチ
k = 2 * np.pi / wavelength

z_rs = 5e-3                  # 平面物体からRS面までの距離 (5 mm)

# 画像パラメータ
I_views = 64                
J_views = 64                
M_px = 16                     
N_px = 16                     

# 元のピクセル解像度 (1024 x 1024)
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> Original RS Resolution: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\Grass_16px_multiview_output_fullparallax_64x64_z0005"
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*")))

if len(image_paths) == 0:
    raise FileNotFoundError("画像ファイルが見つかりません。")

u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

for path in image_paths:
    filename = os.path.basename(path)
    match = re.search(r"view_(\d+)_(\d+)", filename)
    if not match: continue
        
    iy, ix = int(match.group(1)), int(match.group(2))
    if iy >= J_views or ix >= I_views: continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
        
    img = cv2.flip(img, 0).astype(np.float32)
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
# 2.5 RS平面のゼロパディング
# =====================================================================
print("--> Applying Zero-Padding to RS plane (2N x 2N)...")
N_x_pad, N_y_pad = 2 * N_x, 2 * N_y
u_RS_padded = np.zeros((N_y_pad, N_x_pad), dtype=np.complex64)
y_offset, x_offset = N_y // 2, N_x // 2
u_RS_padded[y_offset : y_offset + N_y, x_offset : x_offset + N_x] = u_RS
del u_RS 

# =====================================================================
# 3. 帯域制限付き角スペクトル法 (Band-Limited ASM)
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
    
    return sfft.ifft2(np.fft.ifftshift(U_freq), workers=-1)

# =====================================================================
# 4. 関数化 (CGH生成 ＆ Zスキャン保存)
# =====================================================================
slm_size = 1080  
slm_pitch = 8.0e-6 
rec_size = 1080

def generate_reconstruct_and_scan(D_val, output_dir):
    print(f"\n========== Starting process for D = {D_val*1000:.0f} mm ==========")
    z_rs_to_cgh = D_val - z_rs
    
    # RS面からCGH面へ伝搬
    cgh_obj_complex = propagate_asm(u_RS_padded, z_rs_to_cgh, wavelength, pitch)
    
    # 抽出とリサイズ
    cy, cx = N_y_pad // 2, N_x_pad // 2
    cgh_original_area = cgh_obj_complex[cy - N_y//2 : cy + N_y//2, cx - N_x//2 : cx + N_x//2]
    
    real_resized = cv2.resize(np.real(cgh_original_area).astype(np.float32), (slm_size, slm_size), interpolation=cv2.INTER_AREA)
    imag_resized = cv2.resize(np.imag(cgh_original_area).astype(np.float32), (slm_size, slm_size), interpolation=cv2.INTER_AREA)
    cgh_obj_slm = real_resized + 1j * imag_resized
    
   # バイナリ化 (0 or π)
    cgh_phase_binary = np.where(np.cos(np.angle(cgh_obj_slm)) >= 0, 0.0, np.pi)
    slm_8bit = np.round((cgh_phase_binary / (2 * np.pi)) * 255).astype(np.uint8)
    
    # ---------------------------------------------------
    # ▼▼▼ CGHをPNGで保存 ▼▼▼
    os.makedirs(output_dir, exist_ok=True) # ディレクトリがない場合を考慮
    cgh_filename = os.path.join(output_dir, f"cgh_D{D_val*1000:.0f}mm.png")
    slm_8bit_flipped = np.flipud(slm_8bit)  # 上下反転して保存
    cv2.imwrite(cgh_filename, slm_8bit_flipped)
    print(f"--> Saved CGH image: {cgh_filename}")
    # ▲▲▲ 追加ここまで ▲▲▲
    # ---------------------------------------------------

    # 照明波面の作成
    cgh_illuminated = np.zeros((rec_size, rec_size), dtype=np.complex64)

    c_y, c_x = rec_size // 2, rec_size // 2
    slm_phase_reconstructed = (slm_8bit.astype(np.float32) / 255.0) * 2 * np.pi
    cgh_illuminated[c_y - slm_size//2 : c_y + slm_size//2, c_x - slm_size//2 : c_x + slm_size//2] = np.exp(1j * slm_phase_reconstructed)
    
    # スキャンの実行
    os.makedirs(output_dir, exist_ok=True)
    z_scan_list = np.arange(-200e-3, 205e-3, 5e-3)
    
    print(f"--> Scanning from -200mm to +200mm and saving to '{output_dir}'...")
    rec_true = None
    
    for z_rec in z_scan_list:
        img_wave = propagate_asm(cgh_illuminated, z_rec, wavelength, slm_pitch)
        
        raw_rec_intensity = np.abs(img_wave)**2
        if np.max(raw_rec_intensity) > 0:
            raw_rec_intensity /= np.max(raw_rec_intensity)

        rec_intensity = raw_rec_intensity ** 0.5 
        vmax_val = np.percentile(rec_intensity, 99.9)
        if vmax_val > 0:
            rec_intensity = np.clip(rec_intensity / vmax_val, 0.0, 1.0)
        else:
            rec_intensity = np.zeros_like(rec_intensity)

        # 画像保存
        save_img = np.uint8(rec_intensity * 255)
        save_img = np.flipud(save_img)  # 上下反転して保存
        save_img_color = cv2.applyColorMap(save_img, cv2.COLORMAP_INFERNO)
        filename = f"z_{z_rec*1000:+04.0f}mm.png"
        cv2.imwrite(os.path.join(output_dir, filename), save_img_color)
        
        # プロット用に虚像(True Image)の面を保持
        if np.isclose(z_rec, -D_val):
            rec_true = rec_intensity.copy()
            
    # もしスキャン間隔にジャストピント位置が含まれていなかった場合の保険
    if rec_true is None:
        img_wave_true = propagate_asm(cgh_illuminated, -D_val, wavelength, slm_pitch)
        raw_t = np.abs(img_wave_true)**2
        if np.max(raw_t) > 0: raw_t /= np.max(raw_t)
        rec_t = raw_t ** 0.5
        vmax_val_t = np.percentile(rec_t, 99.9)
        rec_true = np.clip(rec_t / vmax_val_t, 0.0, 1.0) if vmax_val_t > 0 else np.zeros_like(rec_t)

    print("--> Scan complete.")
    return cgh_phase_binary, rec_true

# =====================================================================
# 5. 計算実行
# =====================================================================
# D = 10mm
cgh_10, img_10_true = generate_reconstruct_and_scan(10e-3, "z_scan_D10mm")

# D = 100mm
cgh_100, img_100_true = generate_reconstruct_and_scan(100e-3, "z_scan_D100mm")

# =====================================================================
# 6. プロット (3枚並べ)
# =====================================================================
print("\n--> Generating summary figures...")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

slm_extent_mm = [-slm_size*slm_pitch/2*1e3, slm_size*slm_pitch/2*1e3, -slm_size*slm_pitch/2*1e3, slm_size*slm_pitch/2*1e3]
rec_extent_mm = [-rec_size*slm_pitch/2*1e3, rec_size*slm_pitch/2*1e3, -rec_size*slm_pitch/2*1e3, rec_size*slm_pitch/2*1e3]

# ① バイナリCGH (D=10mmのものを使用)
im0 = ax[0].imshow(cgh_10, cmap='gray', extent=slm_extent_mm, origin='lower', vmin=0, vmax=np.pi)
ax[0].set_title("Binary CGH (D=10mm)")
cbar_phase = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
cbar_phase.set_ticks([0, np.pi])
cbar_phase.set_ticklabels(['0', 'π'])

# ② D=10mmの虚像
im1 = ax[1].imshow(img_10_true, cmap='inferno', extent=rec_extent_mm, origin='lower', vmin=0, vmax=1.0)
ax[1].set_title("Virtual Image (D=10mm)")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

# ③ D=100mmの虚像
im2 = ax[2].imshow(img_100_true, cmap='inferno', extent=rec_extent_mm, origin='lower', vmin=0, vmax=1.0)
ax[2].set_title("Virtual Image (D=100mm)")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_xlabel("x [mm]")
    a.set_ylabel("y [mm]")

plt.tight_layout()
plt.show()