import glob
import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft
import sys

# Windows のコンソール (cp932) で表示できない文字があっても停止しないようにする
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# =====================================================================
# 【テスト版】記録時の参照光と再構成時の再生光を 1対1 で対応させた版
#   (元: AngleMux4_RS_MetaHologram.py)
#
#   元の版: 4つのCGHはすべて垂直入射の参照光 R = 1 で記録 (キノフォーム φ = arg(O))。
#           斜めの再生光の傾きはメタ原子格子の迂回位相が担う。
#   この版: 各シーンを、そのシーンを再生する照明と同じ斜めの平面波
#             R_i(x, y) = exp[ j 2π (s_x x + s_y y) / P ]   (sinθ0 = λ/P, 振幅 1)
#           を参照光として記録する (キノフォーム φ_i = arg(O_i R_i*))。
#             West : R = exp(+j2πx/P)  (+θ0, 0)     South: R = exp(+j2πy/P)  (0, +θ0)
#             East : R = exp(-j2πx/P)  (-θ0, 0)     North: R = exp(-j2πy/P)  (0, -θ0)
#           再構成では同じ関数 R_i を、変位したメタ原子の実際の位置 (mP + d) で評価して照射する。
#
#   出力ファイル名には "_refmatch" が付き、元の版の結果は上書きしない。
#   最後に元の版 (垂直参照光) の再生像との相関・差も表示する。
# =====================================================================
# =====================================================================
# 4方向の角度多重メタホログラム (RS法キノフォーム x 4 を2種類のメタ原子の変位にエンコード)
#   参考: Jang et al., Adv. Opt. Mater. 9, 2100678 (2021), Fig. 5
#
#   メタ原子1 (+θ の光を垂直方向へ回折, -θ の光はほぼ回折しない):
#       dx1 = +φ_West  P / 2π  → 照明 (+θ0, 0) で West が再生
#       dy1 = +φ_South P / 2π  → 照明 (0, +θ0) で South が再生
#   メタ原子2 (メタ原子1を180°回転, 角度選択性が逆):
#       dx2 = -φ_East  P / 2π  → 照明 (-θ0, 0) で East が再生 (逆向き照明では位相の符号が反転するため)
#       dy2 = -φ_North P / 2π  → 照明 (0, -θ0) で North が再生
#   
#   メタ原子の角度選択性はベクトル電磁界解析の結果であり、スカラー近似では再現できないため、
#   「メタ原子の種類 x 照明の向き」ごとの回折効率 (論文 Fig.3: 36% / 0.1%) を与える現象論モデルとする。
#
#   再生モデルは AngleMux_RS_MetaHologram.py と同じ:
#     S(f) = Σ_m c_m keep_m exp(j s 2π d_m/P) exp(-j2π f・(r_m + d_m))   (瞳内, テイラー展開 + FFT)
#     → -D へ逆伝搬 + レンズ瞳ローパス (RS_kinoform.py と同じ結像)
# =====================================================================

# =====================================================================
# 1. パラメータ設定 (RS_kinoform.py と共通)
# =====================================================================
# 論文 (Jang et al.) と同じ波長・周期。多視点画像も同じ条件で撮影 (Picture_parallel_633.py)
wavelength = 633e-9          # 光の波長 λ: 633 nm
pitch = 0.7e-6               # CGHのピクセルピッチ = メタ原子の周期 P = 700 nm
k = 2 * np.pi / wavelength

z_rs = 5e-3                  # 平面物体からRS面までの距離 (撮影時のカメラ距離 5 mm と同じ)
# RS面からCGH面までの距離。伝搬は角スペクトル法で厳密に計算するので、距離による像のぼけは生じない。
# ただしピッチ 0.7 µm では物体光が ±sin^-1(λ/2p) = ±27° まで広がり、遠ざけるほど大きく傾いた光線が
# ホログラム (ゼロパディング後 ±11.5 mm) の外へ出て記録されなくなる (= 斜めから見たときの視域が狭くなる)。
#   帯域いっぱい (±27°) を残す目安: 5.7 mm + z tan27° < 11.5 mm → z < 約 11 mm
#   正面から小さな瞳で観察するだけなら、z を大きくしても再生像への影響は小さい。
z_rs_to_cgh = 20e-3
D = z_rs + z_rs_to_cgh       # 平面物体からCGHまでの距離

lens_distance = 200e-3       # CGHからレンズまでの距離
pupil_diameter = 7e-3        # レンズの瞳直径

I_views = 256                # 水平方向の視点数
J_views = 256                # 垂直方向の視点数
M_px = 64                    # 各投影画像の横解像度
N_px = 64                    # 各投影画像の縦解像度

# 4つのシーン (多視点画像フォルダ) と対応する照明 (論文 Fig.5 と同じ割り当て)
scene_root = r"C:\Lab\Meta_letter"
scenes = {
    "West":  os.path.join(scene_root, "West"),    # 照明 (+θ0, 0), メタ原子1 の x 変位
    "South": os.path.join(scene_root, "South"),   # 照明 (0, +θ0), メタ原子1 の y 変位
    "East":  os.path.join(scene_root, "East"),    # 照明 (-θ0, 0), メタ原子2 の x 変位
    "North": os.path.join(scene_root, "North"),   # 照明 (0, -θ0), メタ原子2 の y 変位
}

# =====================================================================
# メタサーフェスのパラメータ
# =====================================================================
P = pitch                    # メタ原子の周期 (CGHの1ピクセル = 1メタ原子)
n_sub = 1.45                 # 基板の屈折率
atom_footprint = 300e-9      # メタ原子 (quadrumer) の外形サイズ [m] (論文: 約300 nm)
disp_step = None             # 変位の量子化ステップ [m] (作製分解能, 例: 5e-9)。None なら量子化なし
taylor_order = 2             # exp(-j2π f・d) のテイラー展開次数

# 空間多重の配置: 'checker' (市松模様), 'rows' (行ごとに交互), 'cols' (列ごとに交互),
#                 'random' (各格子点のメタ原子の種類をランダムに決める。周期性がないため折り返しの
#                           ゴーストは生じず、その代わりに全帯域に広がったノイズになる)
layout = 'checker'
random_seed = 1234           # layout = 'random' の乱数の種

# メタ原子の角度選択性 (垂直方向への回折効率)。
#   RCWA (RCWA_Spectrum.py, 633 nm, P = 700 nm, a-Si 3.19+0.02i) の計算値: 32.7 % (望む向き), 1.8 % (逆向き)
#   論文 Fig.3c,d の値を使う場合: eta_on = 0.36, eta_off = 0.001
eta_on = 0.327
eta_off = 0.018

# 空間多重による折り返し (エイリアシング) 対策:
#   各シーンは半分のメタ原子 (副格子) でサンプリングされるため、副格子で表せない帯域の成分が
#   瞳内に折り返してゴースト像になる。RS面の各要素画像 (= 光線方向の分布) のうち、
#   副格子の帯域 (checker: ひし形 |u|+|v| < 1/2, rows: |v| < 1/2, cols: |u| < 1/2, 帯域端 = 1) の外側を 0 にする。
band_limit_views = False     # layout = 'random' では副格子が無いので自動的に無効
band_margin = 0.9            # 帯域に対する余裕 (1.0 で帯域ちょうど)

rec_oversample = 2           # 再生像のオーバーサンプリング (瞳内スペクトルのゼロパディング倍率)
zoom_range = 8.0             # 拡大表示の範囲 [mm] (±)。RS面は 11.5 mm 角
block_rows = 512             # ブロック処理の行数 (メモリ節約)

# 出力先: このスクリプトと同じフォルダ内 (どこから実行しても同じ場所になる)
# 633 nm / 700 nm 用の出力先 (2 µm 版のキャッシュと混ざらないよう別フォルダ)。
# CGH は RS-CGH 距離で変わるため、距離ごとにフォルダを分ける (古い距離のキャッシュを読み込まないため)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"AngleMux4_RS_output_633_z{z_rs_to_cgh * 1e3:g}mm")
os.makedirs(out_dir, exist_ok=True)

N_x = I_views * M_px
N_y = J_views * N_px
N_x_pad = 2 * N_x
N_y_pad = 2 * N_y
L_x = N_x_pad * P
L_y = N_y_pad * P

theta_air = np.degrees(np.arcsin(wavelength / P))
theta_sub = np.degrees(np.arcsin(wavelength / (n_sub * P)))
print(f"--> Metasurface: {N_x_pad} x {N_y_pad} atoms, P = {P * 1e9:.0f} nm, size {L_x * 1e3:.2f} mm x {L_y * 1e3:.2f} mm")
print(f"--> Illumination angle θ0: {theta_air:.1f} deg (air), {theta_sub:.1f} deg (substrate)")
print(f"--> Layout: {layout}, selectivity: eta_on = {eta_on}, eta_off = {eta_off}")
print(f"--> Object-RS {z_rs * 1e3:.1f} mm, RS-CGH {z_rs_to_cgh * 1e3:.1f} mm, lens {lens_distance * 1e3:.0f} mm, "
      f"pupil {pupil_diameter * 1e3:.1f} mm")

# =====================================================================
# 2. RS法によるキノフォームCGHの作成 (RS_kinoform.py と同じ処理)
# =====================================================================
def propagate_asm_inplace(u, z):
    """帯域制限付き角スペクトル法 (RS_kinoform.py の propagate_asm と同じ伝達関数)。u は上書きされる"""
    Ny, Nx = u.shape
    Lx, Ly = Nx * pitch, Ny * pitch
    fx = sfft.fftfreq(Nx, pitch).astype(np.float32)
    fy = sfft.fftfreq(Ny, pitch).astype(np.float32)
    limit_x = (Lx / 2) / np.sqrt((Lx / 2) ** 2 + z ** 2) / wavelength
    limit_y = (Ly / 2) / np.sqrt((Ly / 2) ** 2 + z ** 2) / wavelength

    U = sfft.fft2(u, workers=-1, overwrite_x=True)
    del u
    for r0 in range(0, Ny, block_rows):
        r1 = min(r0 + block_rows, Ny)
        FY = fy[r0:r1, None]
        sq = 1.0 - (wavelength * fx[None, :]) ** 2 - (wavelength * FY) ** 2
        np.maximum(sq, 0, out=sq)
        H = np.exp(1j * k * z * np.sqrt(sq)).astype(np.complex64)
        H[:, np.abs(fx) > limit_x] = 0
        H[np.abs(fy[r0:r1]) > limit_y, :] = 0
        U[r0:r1] *= H
    return sfft.ifft2(U, workers=-1, overwrite_x=True)


def make_rs_kinoform(image_folder, cache_path, seed):
    """多視点画像から RS 面を作り、CGH 面へ伝搬してキノフォーム位相を得る。
    位相は uint16 (2π/65536 刻み) で .npy に保存し、次回以降は再利用する。"""
    if os.path.exists(cache_path):
        print(f"--> Load cached CGH: {cache_path}")
        return
    rng = np.random.default_rng(seed)
    image_paths = sorted(glob.glob(os.path.join(image_folder, "view_*")))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {image_folder}")

    u = np.zeros((N_y_pad, N_x_pad), dtype=np.complex64)
    y_offset, x_offset = N_y // 2, N_x // 2
    for path in image_paths:
        match = re.search(r"view_(\d+)_(\d+)", os.path.basename(path))
        if not match:
            continue
        iy, ix = int(match.group(1)), int(match.group(2))
        if iy >= J_views or ix >= I_views:
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.flip(img, 0).astype(np.float32)
        if img.shape != (N_px, M_px):
            img = cv2.resize(img, (M_px, N_px), interpolation=cv2.INTER_AREA)
        img = np.sqrt(np.maximum(img, 0))
        if view_mask is not None:
            img = img * view_mask
        complex_light = img * np.exp(1j * rng.uniform(0, 2 * np.pi, size=(N_px, M_px)))
        RS_val = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(complex_light)))
        y0 = y_offset + iy * N_px
        x0 = x_offset + ix * M_px
        u[y0:y0 + N_px, x0:x0 + M_px] = RS_val
    print(f"--> RS plane is created from {image_folder}")

    print(f"--> Propagating RS plane to CGH plane (z = {z_rs_to_cgh * 1e3:.1f} mm)...")
    u = propagate_asm_inplace(u, z_rs_to_cgh)

    q = np.empty((N_y_pad, N_x_pad), dtype=np.uint16)
    for r0 in range(0, N_y_pad, block_rows):
        r1 = min(r0 + block_rows, N_y_pad)
        ph = np.mod(np.angle(u[r0:r1]), 2 * np.pi)
        q[r0:r1] = (np.round(ph * (65536 / (2 * np.pi))).astype(np.uint32) % 65536).astype(np.uint16)
    del u
    np.save(cache_path, q)
    print(f"--> Kinoform CGH saved: {cache_path}")


def make_view_mask():
    """要素画像の画素 (u, v) (中心 0, 帯域端 ±1) のうち副格子で表せる帯域だけを残すマスク"""
    if not band_limit_views or layout == 'random':
        return None
    u = (np.arange(M_px) - M_px // 2) / (M_px / 2)
    v = (np.arange(N_px) - N_px // 2) / (N_px / 2)
    U, V = np.meshgrid(u, v)
    if layout == 'checker':
        m = np.abs(U) + np.abs(V) < band_margin
    elif layout == 'rows':
        m = np.abs(V) < 0.5 * band_margin
    elif layout == 'cols':
        m = np.abs(U) < 0.5 * band_margin
    else:
        raise ValueError(layout)
    return m.astype(np.float32)


view_mask = make_view_mask()
bl_tag = f"bl-{layout}{band_margin:g}" if view_mask is not None else "nobl"
print(f"--> View band limit: {bl_tag}")

cache = {}
for s, (name, folder) in enumerate(scenes.items()):
    cache[name] = os.path.join(out_dir, f"cgh_{name}_{bl_tag}.npy")
    make_rs_kinoform(folder, cache[name], seed=s)
cgh_normal = {name: np.load(path, mmap_mode='r') for name, path in cache.items()}   # 垂直参照光 (元の版)

# =====================================================================
# 2.5 参照光 (記録時) = 再生光 (再構成時) の定義
#     s = (s_x, s_y): 平面波の横方向空間周波数 (単位 1/P)。sinθ0 = λ/P のとき |s| = 1
# =====================================================================
ref_dirs = {
    "West":  (+1, 0),   # (+θ0, 0)
    "South": (0, +1),   # (0, +θ0)
    "East":  (-1, 0),   # (-θ0, 0)
    "North": (0, -1),   # (0, -θ0)
}


def reference_phase_cycles(name, X, Y):
    """参照光 R = exp(j2π (s_x X + s_y Y)/P) の位相 [周期 (= 2π 単位)]。X, Y [m] (float64 で計算)"""
    sx, sy = ref_dirs[name]
    return (sx * np.asarray(X, dtype=np.float64) + sy * np.asarray(Y, dtype=np.float64)) / P


def record_with_reference(name, src, dst_path):
    """キノフォーム φ = arg(O R*) を記録する。
    src は arg(O) (垂直参照光で記録したキノフォーム)。arg(O R*) = arg(O) - arg(R) は画素ごとの演算なので、
    物体光 O から記録し直すのと厳密に同じ結果になる。R は CGH の画素中心 x = j p, y = i p で評価する。"""
    if os.path.exists(dst_path):
        print(f"--> Load cached CGH (oblique reference): {dst_path}")
        return
    q = np.lib.format.open_memmap(dst_path, mode='w+', dtype=np.uint16, shape=(N_y_pad, N_x_pad))
    x = np.arange(N_x_pad) * pitch
    n_changed = 0
    for r0 in range(0, N_y_pad, block_rows):
        r1 = min(r0 + block_rows, N_y_pad)
        y = (np.arange(r0, r1) * pitch)[:, None]
        cyc = reference_phase_cycles(name, x[None, :], y)
        ref_q = np.round(np.mod(cyc, 1.0) * 65536).astype(np.int64) % 65536
        new = (src[r0:r1].astype(np.int64) - ref_q) % 65536
        n_changed += int(np.count_nonzero(new != src[r0:r1]))
        q[r0:r1] = new.astype(np.uint16)
    q.flush()
    del q
    print(f"--> Recorded {name} with oblique reference s = {ref_dirs[name]}: "
          f"{n_changed} / {N_x_pad * N_y_pad} pixels differ from the normal-reference CGH")


cgh = {}
for name in scenes:
    path = os.path.join(out_dir, f"cgh_{name}_{bl_tag}_refmatch.npy")
    record_with_reference(name, cgh_normal[name], path)
    cgh[name] = np.load(path, mmap_mode='r')

# =====================================================================
# 3. 2種類のメタ原子の配置と変位 (位相を uint16 のまま合成して保存)
#    type 1: (dx, dy) = (+φ_W, +φ_S) P/2π,  type 2: (dx, dy) = (-φ_E, -φ_N) P/2π
# =====================================================================
if layout == 'random':
    # ランダム配置 (各格子点で 1/2 の確率でメタ原子1)。ブロックごとに乱数の種を固定して再現性を持たせる
    type_map = np.empty((N_y_pad, N_x_pad), dtype=bool)
    for r0 in range(0, N_y_pad, block_rows):
        r1 = min(r0 + block_rows, N_y_pad)
        type_map[r0:r1] = np.random.default_rng([random_seed, r0]).random((r1 - r0, N_x_pad), dtype=np.float32) < 0.5
    print(f"--> Random layout: atom 1 ratio = {type_map.mean():.4f}")


def atom_type_block(r0, r1):
    """True: メタ原子1, False: メタ原子2"""
    if layout == 'random':
        return type_map[r0:r1]
    ii = np.arange(r0, r1)[:, None]
    jj = np.arange(N_x_pad)[None, :]
    if layout == 'checker':
        return (ii + jj) % 2 == 0
    if layout == 'rows':
        return np.broadcast_to(ii % 2 == 0, (r1 - r0, N_x_pad))
    if layout == 'cols':
        return np.broadcast_to(jj % 2 == 0, (r1 - r0, N_x_pad))
    raise ValueError(layout)


phase_x_path = os.path.join(out_dir, f"phase_x_{layout}_{bl_tag}_refmatch.npy")
phase_y_path = os.path.join(out_dir, f"phase_y_{layout}_{bl_tag}_refmatch.npy")
if not (os.path.exists(phase_x_path) and os.path.exists(phase_y_path)):
    print("--> Composing displacement maps of two meta-atom types...")
    px = np.lib.format.open_memmap(phase_x_path, mode='w+', dtype=np.uint16, shape=(N_y_pad, N_x_pad))
    py = np.lib.format.open_memmap(phase_y_path, mode='w+', dtype=np.uint16, shape=(N_y_pad, N_x_pad))
    for r0 in range(0, N_y_pad, block_rows):
        r1 = min(r0 + block_rows, N_y_pad)
        t1 = atom_type_block(r0, r1)
        # uint16 の -q は 2π - φ (mod 2π) に等しい
        neg_E = (-cgh["East"][r0:r1].astype(np.int32)) % 65536
        neg_N = (-cgh["North"][r0:r1].astype(np.int32)) % 65536
        px[r0:r1] = np.where(t1, cgh["West"][r0:r1], neg_E).astype(np.uint16)
        py[r0:r1] = np.where(t1, cgh["South"][r0:r1], neg_N).astype(np.uint16)
    px.flush(); py.flush()
    del px, py
phase_x = np.load(phase_x_path, mmap_mode='r')
phase_y = np.load(phase_y_path, mmap_mode='r')


def load_displacement(q_block):
    """uint16 位相 → 位相 φ ∈ [-π, π) → 変位 d = φ P / 2π"""
    ph = q_block.astype(np.float32) * np.float32(2 * np.pi / 65536)
    ph[ph >= np.pi] -= np.float32(2 * np.pi)
    d = ph * np.float32(P / (2 * np.pi))
    if disp_step is not None:
        d = (np.round(d / disp_step) * disp_step).astype(np.float32)
    return d

# =====================================================================
# 4. 重なったメタ原子の削除 (ラスター順に、既に残した原子と重なる原子を削除)
# =====================================================================
print("--> Checking overlapped meta-atoms...")
w = atom_footprint
keep = np.zeros((N_y_pad, N_x_pad), dtype=bool)
dx_prev = dy_prev = keep_prev = None
for r0 in range(0, N_y_pad, block_rows):
    r1 = min(r0 + block_rows, N_y_pad)
    dx_blk = load_displacement(phase_x[r0:r1])
    dy_blk = load_displacement(phase_y[r0:r1])
    for r in range(r1 - r0):
        dx, dy = dx_blk[r], dy_blk[r]
        cand = np.ones(N_x_pad, dtype=bool)
        if keep_prev is not None:
            for dj in (-1, 0, 1):
                nb_dx = np.roll(dx_prev, -dj)
                nb_dy = np.roll(dy_prev, -dj)
                nb_keep = np.roll(keep_prev, -dj)
                if dj == 1:
                    nb_keep[-1] = False
                if dj == -1:
                    nb_keep[0] = False
                ov = (np.abs(dj * P + nb_dx - dx) < w) & (np.abs(-P + nb_dy - dy) < w) & nb_keep
                cand &= ~ov
        ov_left = np.zeros(N_x_pad, dtype=bool)
        ov_left[1:] = (np.abs(P + dx[1:] - dx[:-1]) < w) & (np.abs(dy[1:] - dy[:-1]) < w)
        kp = cand.copy()
        while True:
            left_keep = np.zeros_like(kp)
            left_keep[1:] = kp[:-1]
            new = cand & ~(ov_left & left_keep)
            if np.array_equal(new, kp):
                break
            kp = new
        keep[r0 + r] = kp
        dx_prev, dy_prev, keep_prev = dx, dy, kp
erased_ratio = 1 - keep.mean()
print(f"--> Erased overlapped meta-atoms: {100 * erased_ratio:.2f} %")

# =====================================================================
# 5. 各照明に対する垂直方向付近の回折光スペクトル (瞳内)
# =====================================================================
NA = (pupil_diameter / 2) / (D + lens_distance)
f_c = NA / wavelength
kc_x = int(np.ceil(f_c * L_x))
kc_y = int(np.ceil(f_c * L_y))
kx_idx = np.arange(-kc_x, kc_x + 1)
ky_idx = np.arange(-kc_y, kc_y + 1)
fx_c = (kx_idx / L_x).astype(np.float32)
fy_c = (ky_idx / L_y).astype(np.float32)
FXc, FYc = np.meshgrid(fx_c, fy_c)
print(f"--> Pupil NA = {NA:.4f}, spectrum inside pupil: {len(kx_idx)} x {len(ky_idx)} samples")

terms = [(p, n - p) for n in range(taylor_order + 1) for p in range(n, -1, -1)]

# 照明: 名前 → (変位の軸, 向き)。対応する再生シーン
illuminations = {
    "(+θ0, 0)": ('x', +1, "West"),
    "(0, +θ0)": ('y', +1, "South"),
    "(-θ0, 0)": ('x', -1, "East"),
    "(0, -θ0)": ('y', -1, "North"),
}


def spectrum_2d(fill_block, names):
    """fill_block(r0, r1) → {name: 行ブロックの複素振幅}。行方向FFT→瞳内の列だけ残す→列方向FFT"""
    T = {nm: np.empty((N_y_pad, len(kx_idx)), dtype=np.complex64) for nm in names}
    cols = kx_idx % N_x_pad
    for r0 in range(0, N_y_pad, block_rows):
        r1 = min(r0 + block_rows, N_y_pad)
        for nm, blk in fill_block(r0, r1):
            T[nm][r0:r1] = sfft.fft(blk, axis=1, workers=-1)[:, cols]
    rows = ky_idx % N_y_pad
    return {nm: sfft.fft(T.pop(nm), axis=0, workers=-1)[rows, :] for nm in names}


ill_to_scene = {('x', +1): "West", ('y', +1): "South", ('x', -1): "East", ('y', -1): "North"}


def metasurface_spectrum(axis, sign):
    """照明 (axis, sign) で垂直方向へ回折する光の瞳内スペクトル (角度選択性・削除・変位誤差を含む)"""
    # メタ原子1 は +θ, メタ原子2 は -θ を効率よく回折する
    c1 = np.float32(np.sqrt(eta_on if sign > 0 else eta_off))
    c2 = np.float32(np.sqrt(eta_off if sign > 0 else eta_on))

    def fill(r0, r1):
        dx = load_displacement(phase_x[r0:r1])
        dy = load_displacement(phase_y[r0:r1])
        d_main = dx if axis == 'x' else dy
        coef = np.where(atom_type_block(r0, r1), c1, c2).astype(np.float32) * keep[r0:r1]
        # 再生光 = 記録時と同じ参照光 R を、変位したメタ原子の実際の位置 (X, Y) = (jP + dx, iP + dy) で評価
        X = np.arange(N_x_pad)[None, :] * P + dx.astype(np.float64)
        Y = np.arange(r0, r1)[:, None] * P + dy.astype(np.float64)
        cyc = np.mod(reference_phase_cycles(ill_to_scene[(axis, sign)], X, Y), 1.0)
        a = (coef * np.exp(1j * 2 * np.pi * cyc)).astype(np.complex64)
        for (p, q) in terms:
            wgt = a
            if p > 0:
                wgt = wgt * dx ** p
            if q > 0:
                wgt = wgt * dy ** q
            yield (p, q), wgt

    T = spectrum_2d(fill, terms)
    S = np.zeros((len(ky_idx), len(kx_idx)), dtype=np.complex64)
    for (p, q) in terms:
        coef = (-2j * np.pi) ** (p + q) / (math.factorial(p) * math.factorial(q))
        S += (coef * FXc ** p * FYc ** q * T[(p, q)]).astype(np.complex64)
    return S


def ideal_spectrum(name):
    """理想: 斜めの参照光 R で記録した普通の (メタサーフェスでない) キノフォームに、同じ R を当てて再生。
    透過光 exp(jφ) R を画素中心で評価する。振幅は sqrt(eta_on) 倍"""
    x = np.arange(N_x_pad) * pitch

    def fill(r0, r1):
        ph = cgh[name][r0:r1].astype(np.float64) * (2 * np.pi / 65536)
        y = (np.arange(r0, r1) * pitch)[:, None]
        ph = ph + 2 * np.pi * np.mod(reference_phase_cycles(name, x[None, :], y), 1.0)
        yield "ideal", (np.float32(np.sqrt(eta_on)) * np.exp(1j * ph)).astype(np.complex64)
    return spectrum_2d(fill, ["ideal"])["ideal"]

# =====================================================================
# 6. レンズによる結像 (RS_kinoform.py と同じ: -D へ逆伝搬 + 瞳フィルタ)
# =====================================================================
Nrx = rec_oversample * len(kx_idx)
Nry = rec_oversample * len(ky_idx)
sq = 1.0 - (wavelength * FXc) ** 2 - (wavelength * FYc) ** 2
H_back = np.exp(1j * k * (-D) * np.sqrt(np.maximum(sq, 0))).astype(np.complex64)
pupil = (FXc ** 2 + FYc ** 2) <= f_c ** 2


def image_from_spectrum(S):
    O = np.zeros((Nry, Nrx), dtype=np.complex64)
    O[np.ix_(ky_idx % Nry, kx_idx % Nrx)] = S * H_back * pupil
    img = sfft.ifft2(O, workers=-1)
    return (np.abs(img) ** 2).astype(np.float32)


ideal = {}
for name in scenes:
    print(f"--> Ideal kinoform reconstruction: {name} ...")
    ideal[name] = image_from_spectrum(ideal_spectrum(name))

images = {}
for ill, (axis, sign, name) in illuminations.items():
    print(f"--> Illumination {ill} (expected: {name}) ...")
    images[ill] = image_from_spectrum(metasurface_spectrum(axis, sign))
    np.save(os.path.join(out_dir, f"rec_{name}_{layout}_{bl_tag}_refmatch.npy"), images[ill])

# =====================================================================
# 7. 評価 (理想像との相関行列)
# =====================================================================
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

names = list(scenes.keys())
lines = []
lines.append("=== Correlation with ideal kinoform reconstructions ===")
lines.append(f"{'':22s}" + "".join(f"{n:>9s}" for n in names) + "   peak/ideal")
lines.append(f"{'(baseline) ideal ' + '':22s}")
for n1 in names:
    lines.append(f"{'  ideal ' + n1:22s}" + "".join(f"{corr(ideal[n1], ideal[n2]):+9.3f}" for n2 in names))
for ill, (_, _, name) in illuminations.items():
    I = images[ill]
    lines.append(f"{ill + ' -> ' + name:22s}" + "".join(f"{corr(I, ideal[n2]):+9.3f}" for n2 in names)
                 + f"   {I.max() / ideal[name].max():.3f}")
lines.append(f"erased atoms: {100 * erased_ratio:.2f} %, layout: {layout}, band limit: {bl_tag}, eta_on/eta_off: {eta_on}/{eta_off}")
lines.append("")
lines.append("=== Comparison with the normal-reference version (AngleMux4_RS_MetaHologram.py) ===")
for ill, (_, _, name) in illuminations.items():
    old_path = os.path.join(out_dir, f"rec_{name}_{layout}_{bl_tag}.npy")
    if os.path.exists(old_path):
        old = np.load(old_path)
        I = images[ill]
        rel = float(np.abs(I - old).max() / old.max())
        lines.append(f"{ill + ' -> ' + name:22s} corr(new, old) = {corr(I, old):+.6f}, max|new-old|/max(old) = {rel:.2e}")
    else:
        lines.append(f"{ill + ' -> ' + name:22s} (no result of the original version: {old_path})")
report = "\n".join(lines)
print("\n" + report)
with open(os.path.join(out_dir, f"report_{layout}_{bl_tag}_refmatch.txt"), "w", encoding="utf-8") as fp:
    fp.write(report + "\n")

# =====================================================================
# 8. 描画
# =====================================================================
extent_mm = [-L_x / 2 * 1e3, L_x / 2 * 1e3, -L_y / 2 * 1e3, L_y / 2 * 1e3]
zr = min(zoom_range, L_x / 2 * 1e3, L_y / 2 * 1e3)
I_ref = max(v.max() for v in ideal.values())


def enhance(I):
    J = I ** 0.5
    return np.clip(J / np.percentile(J, 99.9), 0, 1)


fig, ax = plt.subplots(3, 4, figsize=(20, 15))
for c, (ill, (_, _, name)) in enumerate(illuminations.items()):
    I = images[ill]
    ax[0, c].imshow(enhance(ideal[name]), cmap='inferno', extent=extent_mm, origin='lower', vmin=0, vmax=1)
    ax[0, c].set_title(f"Ideal kinoform: {name}\n(Zoomed, Enhanced)")
    ax[1, c].imshow(enhance(I), cmap='inferno', extent=extent_mm, origin='lower', vmin=0, vmax=1)
    ax[1, c].set_title(f"Metasurface, illumination {ill}\n(Zoomed, Enhanced)")
    im = ax[2, c].imshow(I / I_ref, cmap='inferno', extent=extent_mm, origin='lower', vmin=0, vmax=1)
    ax[2, c].set_title(f"Metasurface, illumination {ill}\n(Raw, normalized to ideal peak)")
    fig.colorbar(im, ax=ax[2, c], fraction=0.046, pad=0.04)
    for r in range(3):
        ax[r, c].set_xlim(-zr, zr); ax[r, c].set_ylim(-zr, zr)
        ax[r, c].set_xlabel("x [mm]"); ax[r, c].set_ylabel("y [mm]")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"reconstruction_{layout}_{bl_tag}_refmatch.png"), dpi=150)

# メタ原子配置 (中央 8x8 セル): 黒 = メタ原子1, 青 = メタ原子2 (180°回転), × = 削除
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
n_show = 8
c0y, c0x = N_y_pad // 2, N_x_pad // 2
sdx = load_displacement(phase_x[c0y:c0y + n_show])[:, c0x:c0x + n_show]
sdy = load_displacement(phase_y[c0y:c0y + n_show])[:, c0x:c0x + n_show]
skp = keep[c0y:c0y + n_show, c0x:c0x + n_show]
st1 = atom_type_block(c0y, c0y + n_show)[:, c0x:c0x + n_show]
for i in range(n_show):
    for j in range(n_show):
        cx = (j + 0.5) * P + sdx[i, j]
        cy = (i + 0.5) * P + sdy[i, j]
        if skp[i, j]:
            ax[0].add_patch(plt.Rectangle(((cx - w / 2) * 1e6, (cy - w / 2) * 1e6), w * 1e6, w * 1e6,
                                          color='k' if st1[i, j] else 'tab:blue'))
        else:
            ax[0].plot(cx * 1e6, cy * 1e6, 'rx')
for kk in range(n_show + 1):
    ax[0].axhline(kk * P * 1e6, color='gray', lw=0.4)
    ax[0].axvline(kk * P * 1e6, color='gray', lw=0.4)
ax[0].set_xlim(0, n_show * P * 1e6); ax[0].set_ylim(0, n_show * P * 1e6); ax[0].set_aspect('equal')
ax[0].set_title(f"Meta-atom layout ({layout})\nblack: atom 1 (W/S), blue: atom 2 (E/N)")
ax[0].set_xlabel("x [µm]"); ax[0].set_ylabel("y [µm]")
skip = max(1, N_x_pad // 2048)
ax[1].imshow(keep[::skip, ::skip], cmap='gray', extent=extent_mm, origin='lower', interpolation='nearest')
ax[1].set_title(f"Kept atoms (erased {100 * erased_ratio:.2f} %)"); ax[1].set_xlabel("x [mm]")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"design_{layout}_{bl_tag}_refmatch.png"), dpi=150)
print(f"\n--> Figures saved to '{out_dir}'")
if os.environ.get('NO_SHOW') is None:
    plt.show()
