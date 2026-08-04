import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class OpenHoloLF_Python:
    """OpenHolo (ophLF) の処理ロジックを完全再現したPythonクラス"""
    
    def __init__(self, wavelength=633e-9, pitch=2.0e-6, z_rs_to_cgh=0.015):
        self.lambda_ = wavelength
        self.pitch = pitch
        self.z_rs_to_cgh = z_rs_to_cgh
        self.k = 2 * np.pi / self.lambda_
        
        self.num_image = (128, 128)      # 視点数 (nx, ny)
        self.resol_image = (64, 64)      # 各視点画像の解像度 (px, py)
        
        self.u_RS = None
        self.cgh_complex = None
        self.cgh_phase = None
        self.reconstructed_view_0deg = None

    def convert_lf_to_complex_field(self, image_folder, ext="png"):
        """1. 多視点画像からRS平面波面を作成 (convertLF2ComplexField)"""
        I_views, J_views = self.num_image
        M_px, N_px = self.resol_image
        
        N_x = I_views * M_px
        N_y = J_views * N_px
        
        self.u_RS = np.zeros((N_y, N_x), dtype=np.complex64)
        
        image_paths = sorted(glob.glob(os.path.join(image_folder, f"*.{ext}")))
        if not image_paths:
            raise FileNotFoundError("指定フォルダに画像が見つかりません。パスを確認してください。")

        print("--> 1/4: 多視点画像 (Light Field) を読み込み RS平面波面を構成中...")
        
        for path in image_paths:
            filename = os.path.basename(path)
            parts = filename.replace(f".{ext}", "").split("_")
            iy, ix = int(parts[1]), int(parts[2])

            if iy >= J_views or ix >= I_views:
                continue

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if img.shape != (N_px, M_px):
                img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)

            amp = np.sqrt(np.maximum(img, 0.0))
            random_phase = np.random.uniform(0, 2 * np.pi, size=(N_px, M_px))
            complex_light = amp * np.exp(1j * random_phase)

            # 各要素画像ブロックに対する 2D-FFT
            RS_val = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_light)))

            y_start, y_end = iy * N_px, (iy + 1) * N_px
            x_start, x_end = ix * M_px, (ix + 1) * M_px

            self.u_RS[y_start:y_end, x_start:x_end] = RS_val
            
        print("    RS平面波面の構成完了。")

    def propagate_asm(self, u_in, z):
        """角スペクトル法 (ASM) による伝搬関数"""
        Ny, Nx = u_in.shape

        dfx = 1.0 / (Nx * self.pitch)
        dfy = 1.0 / (Ny * self.pitch)

        fx = (np.arange(Nx) - Nx // 2) * dfx
        fy = (np.arange(Ny) - Ny // 2) * dfy
        FX, FY = np.meshgrid(fx, fy)

        sq = 1.0 - (self.lambda_ * FX)**2 - (self.lambda_ * FY)**2
        sq[sq < 0] = 0.0
        H = np.exp(1j * self.k * z * np.sqrt(sq))

        U_freq = np.fft.fftshift(np.fft.fft2(u_in))
        U_prop_freq = U_freq * H
        u_out = np.fft.ifft2(np.fft.ifftshift(U_prop_freq))

        return u_out

    def generate_hologram(self):
        """2. RS平面からCGH平面へ伝搬して位相ホログラムを生成 (generateHologram)"""
        print(f"--> 2/4: CGH平面へ伝搬中 (z = {self.z_rs_to_cgh*1e3:.1f} mm)...")
        
        self.cgh_complex = self.propagate_asm(self.u_RS, self.z_rs_to_cgh)

        # 実部と虚部から偏角 (位相 rad) を抽出し、純粋な位相ホログラムを作成
        self.cgh_phase = np.angle(self.cgh_complex)
        print("    位相ホログラム (Phase-Only CGH) の生成完了。")

    def reconstruct_view(self, target_angle_deg=(0, 0)):
        """3. 位相ホログラムから焦点面へ逆伝搬し、特定の視点（デフォルト0°正面）の再生像を抽出"""
        print("--> 3/4: 位相ホログラムから焦点面(RS面)へ逆伝搬中...")
        
        # 1. 振幅を1に固定した Phase-Only CGH 波面を作成
        cgh_phase_only = np.exp(1j * self.cgh_phase)
        
        # 2. CGH面からRS面(-z)へ逆伝搬
        rs_rec_complex = self.propagate_asm(cgh_phase_only, -self.z_rs_to_cgh)
        
        print("--> 4/4: 再生波面から 0° 視点の要素画像をデコード抽出中...")
        
        I_views, J_views = self.num_image
        M_px, N_px = self.resol_image
        
        self.reconstructed_view_0deg = np.zeros((J_views, I_views), dtype=np.float32)

        # 3. 各要素ブロックから指定角度（中心画素=0°）成分を切り出す
        for iy in range(J_views):
            for ix in range(I_views):
                y_start = iy * N_px
                x_start = ix * M_px
                
                block = rs_rec_complex[y_start:y_start+N_px, x_start:x_start+M_px]
                
                # IFFT で空間画像へ戻す
                img_rec = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(block))))**2
                
                # 0°視点（画像中央）の光強度を取得
                self.reconstructed_view_0deg[iy, ix] = img_rec[N_px // 2, M_px // 2]
                
        print("    像再生および抽出処理完了。")

# =====================================================================
# メイン処理 (実行パート)
# =====================================================================
if __name__ == "__main__":
    # パラメータ設定
    folder = r"C:\Lab\Sphere_multiview_output_fullparallax_256x256_z0020"
    
    # 1. クラスの初期化
    oph = OpenHoloLF_Python(wavelength=633e-9, pitch=2.0e-6, z_rs_to_cgh=0.015)
    
    # 2. CGH生成パイプライン実行
    oph.convert_lf_to_complex_field(folder, ext="png")
    oph.generate_hologram()
    
    # 3. 再生像の切り出し計算
    oph.reconstruct_view(target_angle_deg=(0, 0))
    
    # 4. 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 左: 再生像 (0° 正面視点) ---
    im0 = axes[0].imshow(oph.reconstructed_view_0deg, cmap='gray', origin='lower')
    axes[0].set_title("Reconstructed View Image (0° Direction)", fontsize=12)
    axes[0].set_xlabel("x (View Index)", fontsize=10)
    axes[0].set_ylabel("y (View Index)", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # --- 右: 生成された位相ホログラム ---
    N_x = oph.num_image[0] * oph.resol_image[0]
    extent_mm = [-N_x*oph.pitch/2*1e3, N_x*oph.pitch/2*1e3, -N_x*oph.pitch/2*1e3, N_x*oph.pitch/2*1e3]
    
    im1 = axes[1].imshow(oph.cgh_phase, cmap='twilight', extent=extent_mm, origin='lower', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Phase-Only CGH Data (at SLM Plane)", fontsize=12)
    axes[1].set_xlabel("x [mm]", fontsize=10)
    axes[1].set_ylabel("y [mm]", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()