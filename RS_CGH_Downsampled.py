import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.fft as sfft
import gc  # メモリ解放用

# =====================================================================
# 1. パラメータ設定
# =====================================================================
wavelength = 532e-9          # 光の波長 λ: 532 nm
pitch = 8.0e-6               # CGH平面の初期ピクセルピッチ (8 um)

z_rs = 5e-3                  # 平面物体からRS面までの距離 (5 mm)

# 画像パラメータ (256視点 x 64px)
I_views = 256                
J_views = 256                
M_px = 64                    
N_px = 64                    

# 元のピクセル解像度 (16384 x 16384)
N_x = I_views * M_px
N_y = J_views * N_px

print(f"--> Original RS Resolution: {N_x} x {N_y} ({N_x * pitch * 1e3:.2f} mm x {N_y * pitch * 1e3:.2f} mm)")

# =====================================================================
# 2. 多視点画像群の読み込みとRS平面波面の計算
# =====================================================================
image_folder = r"C:\Lab\Dice_64px_multiview_output_fullparallax_256x256_z0005" # フォルダ名適宜変更
image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*")))

if len(image_paths) == 0:
    raise FileNotFoundError("画像ファイルが見つかりません。パスを確認してください。")

u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

print("--> Constructing RS plane...")
for path in image_paths:
    filename = os.path.basename(path)
    match = re.search(r"view_(\d+)_(\d+)", filename)
    if not match: continue
        
    iy, ix = int(match.group(1)), int(match.group(2))
    if iy >= J_views or ix >= I_views: continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
        
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
# 2.5 RS平面のゼロパディング (32768 x 32768)
# =====================================================================
print("--> Applying Zero-Padding to RS plane (2N x 2N)...")
N_x_pad, N_y_pad = 2 * N_x, 2 * N_y
u_RS_padded = np.zeros((N_y_pad, N_x_pad), dtype=np.complex64)
y_offset, x_offset = N_y // 2, N_x // 2
u_RS_padded[y_offset : y_offset + N_y, x_offset : x_offset + N_x] = u_RS

del u_RS # 大容量メモリの解放
gc.collect()

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
    
    H = np.exp(1j * (2 * np.pi / wavelength) * z * np.sqrt(sq)).astype(np.complex64)
    limit_x = (Lx / 2) / np.sqrt((Lx / 2)**2 + z**2) / wavelength
    limit_y = (Ly / 2) / np.sqrt((Ly / 2)**2 + z**2) / wavelength
    H[(np.abs(FX) > limit_x) | (np.abs(FY) > limit_y)] = 0.0
    del FX, FY, sq
    gc.collect()
    
    U_freq = np.fft.fftshift(sfft.fft2(u_in, workers=-1))
    U_freq *= H
    del H
    gc.collect()
    
    out = sfft.ifft2(np.fft.ifftshift(U_freq), workers=-1)
    del U_freq
    gc.collect()
    return out

# =====================================================================
# 4. 関数化 (CGH生成 ＆ Zスキャン保存)
# =====================================================================
slm_size = 32768  
slm_pitch = 8.0e-6 
rec_size = 32768

def generate_reconstruct_and_scan(D_val, output_dir):
    print(f"\n========== Starting process for D = {D_val*1000:.0f} mm ==========")
    z_rs_to_cgh = D_val - z_rs
    
    print("--> Propagating massive RS plane to CGH plane...")
    # RS面(32768x32768)からCGH面へ高解像度のまま伝搬
    cgh_obj_complex = propagate_asm(u_RS_padded, z_rs_to_cgh, wavelength, pitch)
    
    print("--> Downsampling full massive plane to SLM resolution...")
    # 中央の抽出を行わず、32768x32768の全領域を1080x1080へと直接リサイズする
    real_resized = cv2.resize(np.real(cgh_obj_complex).astype(np.float32), (slm_size, slm_size), interpolation=cv2.INTER_AREA)
    imag_resized = cv2.resize(np.imag(cgh_obj_complex).astype(np.float32), (slm_size, slm_size), interpolation=cv2.INTER_AREA)
    cgh_obj_slm = real_resized + 1j * imag_resized
    
    del cgh_obj_complex
    gc.collect()
    
    # バイナリ化 (0 or π)
    cgh_phase_binary = np.where(np.cos(np.angle(cgh_obj_slm)) >= 0, 0.0, np.pi)
    slm_8bit = np.round((cgh_phase_binary / (2 * np.pi)) * 255).astype(np.uint8)
    
    # CGHをPNGで保存
    os.makedirs(output_dir, exist_ok=True) 
    cgh_filename = os.path.join(output_dir, f"cgh_D{D_val*1000:.0f}mm.png")
    cv2.imwrite(cgh_filename, slm_8bit)
    print(f"--> Saved CGH image: {cgh_filename}")

    # 照明波面の作成 (以降はSLMサイズでの計算)
    cgh_illuminated = np.zeros((rec_size, rec_size), dtype=np.complex64)
    c_y, c_x = rec_size // 2, rec_size // 2
    slm_phase_reconstructed = (slm_8bit.astype(np.float32) / 255.0) * 2 * np.pi
    cgh_illuminated[c_y - slm_size//2 : c_y + slm_size//2, c_x - slm_size//2 : c_x + slm_size//2] = np.exp(1j * slm_phase_reconstructed)
    
    dir_enhanced = f"{output_dir}_enhanced"
    dir_raw = f"{output_dir}_raw"
    os.makedirs(dir_enhanced, exist_ok=True)
    os.makedirs(dir_raw, exist_ok=True)
    
    z_scan_list = np.arange(-200e-3, 205e-3, 5e-3)
    print(f"--> Scanning from -200mm to +200mm...")
    
    rec_true_raw = None
    rec_true_enhanced = None
    
    for z_rec in z_scan_list:
        img_wave = propagate_asm(cgh_illuminated, z_rec, wavelength, slm_pitch)
        
        raw_rec_intensity = np.abs(img_wave)**2
        if np.max(raw_rec_intensity) > 0:
            raw_rec_intensity /= np.max(raw_rec_intensity)

        # Raw画像保存
        save_img_raw = np.uint8(raw_rec_intensity * 255)
        save_img_raw_color = cv2.applyColorMap(save_img_raw, cv2.COLORMAP_INFERNO)
        filename = f"z_{z_rec*1000:+04.0f}mm.png"
        cv2.imwrite(os.path.join(dir_raw, filename), save_img_raw_color)

        # Enhance画像保存
        rec_intensity = raw_rec_intensity ** 0.5 
        vmax_val = np.percentile(rec_intensity, 99.9)
        if vmax_val > 0:
            rec_intensity = np.clip(rec_intensity / vmax_val, 0.0, 1.0)
        else:
            rec_intensity = np.zeros_like(rec_intensity)

        save_img_enhanced = np.uint8(rec_intensity * 255)
        save_img_enhanced_color = cv2.applyColorMap(save_img_enhanced, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(dir_enhanced, filename), save_img_enhanced_color)
        
        if np.isclose(z_rec, -D_val):
            rec_true_raw = raw_rec_intensity.copy()
            rec_true_enhanced = rec_intensity.copy()
            
    if rec_true_enhanced is None:
        img_wave_true = propagate_asm(cgh_illuminated, -D_val, wavelength, slm_pitch)
        raw_t = np.abs(img_wave_true)**2
        if np.max(raw_t) > 0: raw_t /= np.max(raw_t)
        rec_true_raw = raw_t.copy()
        
        rec_t = raw_t ** 0.5
        vmax_val_t = np.percentile(rec_t, 99.9)
        rec_true_enhanced = np.clip(rec_t / vmax_val_t, 0.0, 1.0) if vmax_val_t > 0 else np.zeros_like(rec_t)

    print("--> Scan complete.")
    # 生データも返すように変更
    return cgh_phase_binary, rec_true_raw, rec_true_enhanced

# =====================================================================
# 5. 計算実行
# =====================================================================
# D = 10mm (3つの戻り値を受け取る)
cgh_10, img_10_true_raw, img_10_true_enh = generate_reconstruct_and_scan(10e-3, "Dice_z_scan_D10mm")

# D = 100mm (今回はプロットしませんが、ファイル保存のために実行・取得はします)
cgh_100, img_100_true_raw, img_100_true_enh = generate_reconstruct_and_scan(100e-3, "Dice_z_scan_D100mm")

# =====================================================================
# 6. プロット (3枚並べ)
# =====================================================================
print("\n--> Generating summary figures...")
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

slm_extent_mm = [-slm_size*slm_pitch/2*1e3, slm_size*slm_pitch/2*1e3, -slm_size*slm_pitch/2*1e3, slm_size*slm_pitch/2*1e3]
rec_extent_mm = [-rec_size*slm_pitch/2*1e3, rec_size*slm_pitch/2*1e3, -rec_size*slm_pitch/2*1e3, rec_size*slm_pitch/2*1e3]

# ① 左：バイナリCGH (D=10mm)
im0 = ax[0].imshow(cgh_10, cmap='gray', extent=slm_extent_mm, origin='lower', vmin=0, vmax=np.pi)
ax[0].set_title("Binary CGH (D=10mm)")
cbar_phase = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
cbar_phase.set_ticks([0, np.pi])
cbar_phase.set_ticklabels(['0', 'π'])

# ② 中央：D=10mmの生データの再生像 (Raw)
im1 = ax[1].imshow(img_10_true_raw, cmap='inferno', extent=rec_extent_mm, origin='lower', vmin=0, vmax=1.0)
ax[1].set_title("Virtual Image Raw (D=10mm)")
fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

# ③ 右：D=10mmの可視化処理した再生像 (Enhanced)
im2 = ax[2].imshow(img_10_true_enh, cmap='inferno', extent=rec_extent_mm, origin='lower', vmin=0, vmax=1.0)
ax[2].set_title("Virtual Image Enhanced (D=10mm)")
fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

for a in ax:
    a.set_xlabel("x [mm]")
    a.set_ylabel("y [mm]")

plt.tight_layout()
plt.show()