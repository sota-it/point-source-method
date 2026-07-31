import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. パラメータ設定
# =====================================================================
wavelength = 633e-9          # 光の波長 λ: 633 nm
pitch = 2e-6                 # ピクセルピッチ (2 μm)
z_rs_to_cgh = 0.05           # RS平面からCGH平面までの距離 (50 mm)
k = 2 * np.pi / wavelength

# 画像パラメータ
I_views = 256                # 水平方向の視点数 (grid_x)
J_views = 256                # 垂直方向の視点数 (grid_y)
M_px = 32                    # 各投影画像の横解像度
N_px = 32                    # 各投影画像の縦解像度

# 全体のピクセル解像度 (8192 x 8192)
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

u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

for path in image_paths:
    filename = os.path.basename(path)
    parts = filename.replace(".png", "").split("_")
    iy, ix = int(parts[1]), int(parts[2])

    if iy >= J_views or ix >= I_views:
        continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if img.shape != (N_px, M_px):
        img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)

    amp = np.sqrt(np.maximum(img, 0)) 
    random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))
    complex_light = amp * np.exp(1j * random_phase)

    # RS平面の空間へ配置 (fftshift適用)
    RS_val = np.fft.fftshift(np.fft.fft2(complex_light))

    y_start, y_end = iy * N_px, (iy + 1) * N_px
    x_start, x_end = ix * M_px, (ix + 1) * M_px
    u_RS[y_start:y_end, x_start:x_end] = RS_val

print("--> RS plane is created.")

# =====================================================================
# 3. 角スペクトル法 (Band-Limited ASM) 伝搬関数定義
# =====================================================================
def propagate_asm(u_in, z, wavelength, pitch):
    """Band-Limited ASM (帯域制限付き角スペクトル法) による自由空間光伝搬"""
    Ny, Nx = u_in.shape
    L_x = Nx * pitch
    L_y = Ny * pitch
    
    dfx = 1.0 / L_x
    dfy = 1.0 / L_y
    
    fx = (np.arange(Nx) - Nx // 2) * dfx
    fy = (np.arange(Ny) - Ny // 2) * dfy
    FX, FY = np.meshgrid(fx, fy)
    
    sq = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    sq[sq < 0] = 0.0
    
    # 基本の伝搬関数 H
    H = np.exp(1j * k * z * np.sqrt(sq))
    
    # 【追加】長距離伝搬 (z > 51mm) での像の増殖(エイリアシング)を防ぐフィルター
    # 伝搬距離 z に応じて、折り返し歪みを起こす高周波成分を 0 にカットする
    if z != 0:
        limit_fx = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * abs(z) / L_x)**2))
        limit_fy = 1.0 / (wavelength * np.sqrt(1.0 + (2.0 * abs(z) / L_y)**2))
        H[(np.abs(FX) > limit_fx) | (np.abs(FY) > limit_fy)] = 0.0

    U_freq = np.fft.fftshift(np.fft.fft2(u_in))
    U_prop_freq = U_freq * H
    u_out = np.fft.ifft2(np.fft.ifftshift(U_prop_freq))
    
    return u_out

# =====================================================================
# 4. RS平面からCGH平面への光波伝搬
# =====================================================================
print(f"--> RS plane to CGH plane ... (distance z = {z_rs_to_cgh*1e3:.1f} mm)...")
cgh_complex = propagate_asm(u_RS, z_rs_to_cgh, wavelength, pitch)

# =====================================================================
# 5. 位相型CGHの生成
# =====================================================================
cgh_phase = np.angle(cgh_complex)
cgh_phase_only_wavefront = np.exp(1j * cgh_phase)
print("--> Phase-only CGH is generated.")

# =====================================================================
# 6. 【追加】仮想レンズ(人間の目)による結像シミュレーション
# =====================================================================
print("--> Simulating the human eye (Imaging Lens)...")

# 座標メッシュの作成
x = (np.arange(N_x) - N_x // 2) * pitch
y = (np.arange(N_y) - N_y // 2) * pitch
X, Y = np.meshgrid(x, y)

# 光学系のパラメータ
d1 = 0.200           # CGH面からレンズ面までの距離 (200 mm)
d2 = 0.050           # レンズ面からセンサー(網膜)までの距離 (50 mm)
pupil_radius = 1.0e-3 # 瞳孔半径 (3.5 mm = 直径 7 mm)

# 焦点距離の計算 (レンズの公式: 1/a + 1/b = 1/f)
# 物体はCGH面からさらに z_rs_to_cgh (50mm) 奥にあるため、レンズからの物体距離 a = 250mm
d_obj_to_lens = z_rs_to_cgh + d1
focal_length = 1.0 / ((1.0 / d_obj_to_lens) + (1.0 / d2))
print(f"    * Calculated lens focal length: {focal_length*1e3:.2f} mm")

# --- Step 6-1: CGH面からレンズ面への伝搬 ---
print(f"    * Propagating from CGH to Lens (z = {d1*1e3:.1f} mm)...")
wave_at_lens_in = propagate_asm(cgh_phase_only_wavefront, d1, wavelength, pitch)

# --- Step 6-2: 瞳孔(アパーチャ)とレンズ位相の適用 ---
# 直径7mmの円形アパーチャ
aperture = np.where((X**2 + Y**2) <= pupil_radius**2, 1.0, 0.0)

# レンズの透過位相関数: exp(-j * k * (x^2 + y^2) / 2f)
lens_phase_func = np.exp(-1j * (k / (2.0 * focal_length)) * (X**2 + Y**2))

# レンズを通過した直後の波面
wave_at_lens_out = wave_at_lens_in * aperture * lens_phase_func

# --- Step 6-3: レンズ面からセンサー(網膜)への伝搬 ---
print(f"    * Propagating from Lens to Sensor/Retina (z = {d2*1e3:.1f} mm)...")
wave_at_sensor = propagate_asm(wave_at_lens_out, d2, wavelength, pitch)

# 強度の算出と正規化
rec_intensity = np.abs(wave_at_sensor) ** 2
max_intensity = np.max(rec_intensity)
rec_intensity /= max_intensity

# =====================================================================
# 7. 結果の可視化
# =====================================================================
extent_mm = [-N_x*pitch/2*1e3, N_x*pitch/2*1e3, -N_y*pitch/2*1e3, N_y*pitch/2*1e3]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# (左) 再生像
# ※現実の凸レンズによる実像結像と同じく「上下左右が反転」するため、そのまま表示します
im0 = axes[0].imshow(rec_intensity, cmap='inferno', extent=extent_mm, origin='lower', vmax=0.1)
axes[0].set_title("Reconstructed View (Through Imaging Lens)")
axes[0].set_xlabel("x [mm]")
axes[0].set_ylabel("y [mm]")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Intensity")

# (右) 位相ホログラム
im1 = axes[1].imshow(cgh_phase, cmap='twilight', extent=extent_mm, origin='lower')
axes[1].set_title("Phase CGH Data")
axes[1].set_xlabel("x [mm]")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Phase [rad]")

plt.tight_layout()
plt.show()