# -*- coding: utf-8 -*-
"""
多視点画像 (512x512 視点, 64px, 2.16um) -> RS平面 -> ASM 伝搬 -> オフアクシス 2 値キノフォーム CGH (0.54um)
-> 左・中央・右から見た再生像シミュレーション

ジオメトリ (観察者側を +z とする):
    物体 (Blender の原点) --5 mm-- RS面 (カメラ面) --15 mm-- CGH面 --> 観察者
    RS面から CGH 面へ +15 mm 伝搬する。物体は CGH 面の 20 mm 奥に虚像として再生される。
    参照光 (= 再生時の照明光) は下から縦方向に THETA_REF だけ傾けて当て、物体光は正面 ±8.79° に出る。
    零次光は上向き THETA_REF、共役像は上向き 2*THETA_REF 付近に分離される。

画素の対応:
    RS 面の 1 画素 (2.16 um) は CGH の 4x4 画素 (0.54 um) に対応する。2.16um で伝搬した物体光を
    0.54um 格子へ補間してから参照光のキャリアを掛け、0.54um 画素ごとに 2 値化する。
    補間は RS_INTERP で選ぶ:
      "fourier": スペクトルのゼロ埋めによる補間 (推奨)。物体光の帯域 ±8.79° をそのまま保つ。
      "nearest": 2.16um の値を 4x4 にそのまま並べる。物体光のコピーが ±17.8°, ±35.6° にでき、
                 共役像のコピーが物体光の帯域に重なってノイズになる。

2 値化 (キノフォーム):
    物体光 O と参照光 R の位相差 arg(O R*) を 0 / π の 2 値に量子化する (cos >= 0 -> 位相 0)。
    物体が CGH から 20 mm 離れているので、物体光の振幅は CGH 面上で広がっており、位相だけでも像が出る。

計算の工夫:
  (1) RS面 32768^2 (8 GiB) を一括で FFT せず、タイル + のりしろに分けて ASM 伝搬する (overlap-save)。
      光は ±8.79° にしか広がらないので、15 mm 伝搬しても広がりは 2.3 mm (1074 px) で、のりしろ内で完結する。
  (2) 補間はタイルの中心部 (+補間用ののりしろ) だけを 4 倍にするので、0.54um の大きな FFT は最小限で済む。
  (3) 参照光のキャリアは 0.54um 格子上で全体座標を使って掛けるので、タイル境界で縞が切れない。
  (4) 再生シミュレーションは、タイルごとに透過率 t を 1 回だけ FFT し、そのスペクトルから
      「照明光 R + 各視点の方向」の周りの窓 (瞳) を切り出して 17.28um の複素振幅に間引く。
      これをつなぎ合わせた後、虚像の位置 (CGH の 20 mm 奥) までまとめて伝搬してピントを合わせる。
      同じ FFT から、ホログラム全体の角度スペクトル (物体光・零次光・共役像の位置) も求める。
"""

import os
import gc
import json
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import scipy.fft as sfft
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =====================================================================
# 1. パラメータ設定
# =====================================================================
RANDOM_SEED = 0

wavelength = 660e-9            # 波長 660 nm
pitch_calc = 2.16e-6           # RS面・伝搬計算のピッチ (撮影時の 2.16 um)
UPSAMPLE = 4                   # CGH ピッチ = pitch_calc / UPSAMPLE = 0.54 um (リソグラフィ装置の画素)
pitch_cgh = pitch_calc / UPSAMPLE
RS_INTERP = "fourier"          # 2.16um -> 0.54um の補間: "fourier" (推奨) / "nearest" (4x4 複製)

I_views, J_views = 512, 512    # 視点数
M_px, N_px = 64, 64            # 1視点あたりの画素数
N_x, N_y = I_views * M_px, J_views * N_px      # 32768 x 32768 (70.78 mm)

z_rs = 5.0e-3                  # 物体 (Blender の原点) -> RS面
D_val = 15.0e-3                # RS面 -> CGH面 (観察者側へ +15 mm)

# 参照光の縦方向の入射角 [deg]。None なら使える範囲の中央を自動で選ぶ。
# 0.54 um では 8.79°〜27.28° が使える。
THETA_REF_DEG = 9.5

# 2 値化
#   "kinoform"       : arg(O R*) を 0 / π に量子化 (2 値キノフォーム)
#   "error_diffusion": h = Re(O R*) を横方向に誤差拡散 (振幅も残す。物体が CGH 面に近いとき用)
BINARIZE = "kinoform"
ED_SCALE = 4.0                 # 誤差拡散のとき、h の標準偏差の何倍を ±1 に対応させるか

# マスクの種類 (シミュレーションでの透過率)
#   "phase"    : bit 1 = 位相 0, bit 0 = 位相 PHASE_DEPTH (キノフォーム)
#   "amplitude": bit 1 = 透過, bit 0 = 遮光
ENCODING = "phase"
PHASE_DEPTH = np.pi            # 製作誤差を見たいときは 0.9*np.pi などにすると零次光が出る

CALC_TILE = 2048               # 伝搬タイルの有効サイズ [2.16um px]
CALC_MARGIN = 1536             # 伝搬ののりしろ [2.16um px] (15 mm で必要量 1074 px + 補間用)
INTERP_MARGIN = 256            # フーリエ補間用ののりしろ [2.16um px]

image_folder = r"C:\Lab\Glassshoes216_64px_multiview_output_fullparallax_512x512_z0005"
OUTPUT_DIR = None              # None なら スクリプトと同じ場所に自動命名
READ_THREADS = 16              # 画像読み込みのスレッド数
PIXEL_THRESHOLD = 2            # この値以下の画素は 0 にする (Blender のディザで黒画像に 1 が散るため)
REUSE_CGH = True               # 保存済み CGH があれば再計算せず、シミュレーションだけ行う
                               # (PIXEL_THRESHOLD が保存時と違うときは作り直す)

# ---- 再生シミュレーション (目は無限遠, 平行光で見る) ----
RUN_SIMULATION = True
VIEWS = [("Left", -8.0), ("Center", 0.0), ("Right", +8.0)]   # 横方向の観察角 [deg]
VIEW_APERTURE_DEG = 1.0        # 瞳の見込み角 (全角) [deg]
FOCUS_ZS = [-20e-3]            # ピントを合わせる位置 (CGH 面から, 負 = 奥)。-20 mm = 物体の原点の虚像
SIM_TILE = 4096 * UPSAMPLE     # シミュレーションのタイル有効サイズ [CGH px]
SIM_MARGIN = 512               # のりしろ [CGH px]
SIM_DOWNSAMPLE = 8 * UPSAMPLE  # 再生像の間引き率 (-> 17.28 um/px)
REFOCUS_PAD = 512              # 虚像へ伝搬するときのゼロ埋め [17.28um px]
SPEC_BINS = 1024               # 角度スペクトル表示の分割数
SHOW_FIGURE = True
# summary 左上に並べる元の物体の画像。Blender で正投影 (Orthographic Scale = RS 面の幅 70.78 mm,
# 中心 X = Z = 0, +Y 向き) で撮ると、再構成像 (0°) と同じ位置・大きさになる。無ければ空欄で表示。
ORIGINAL_IMAGE = os.path.join(image_folder, "original_ortho.png")   # "view_" で始めないこと (視点画像と区別)


# =====================================================================
# 2. 参照光の角度と帯域の確認
# =====================================================================
def deg(s):
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def reference_geometry():
    """sinθ で表した各帯域。s_obj: 物体光の半角, s_nyq: CGH の最大回折角, s_ref: 参照光"""
    s_obj = wavelength / (2 * pitch_calc)
    s_nyq = wavelength / (2 * pitch_cgh)
    s_min, s_max = s_obj, s_nyq - s_obj
    if THETA_REF_DEG is None:
        s_ref = 0.5 * (s_min + s_max)
    else:
        s_ref = np.sin(np.radians(THETA_REF_DEG))

    print("--> Off-axis band check (縦方向, 角度は sinθ を度に換算):")
    print(f"    物体光の角度範囲        : ±{deg(s_obj):.2f}°  (横の視差もこの範囲)")
    print(f"    CGH ピッチ {pitch_cgh*1e6:.2f} um の最大回折角: ±{deg(s_nyq):.2f}°")
    print(f"    参照光の入射角          : {deg(s_ref):.2f}° (下から)")
    print(f"    使える参照光の角度      : {deg(s_min):.2f}° 〜 {deg(s_max):.2f}°")
    print(f"    零次光                  : +{deg(s_ref):.2f}°  (物体光の上端との隙間 {deg(s_ref) - deg(s_obj):+.2f}°)")
    print(f"    共役像                  : +{deg(2*s_ref - s_obj):.2f}° 〜 +{deg(2*s_ref + s_obj):.2f}°")
    print(f"    共役像の画素回折次数    : {deg(2*s_ref + s_obj - 2*s_nyq):.2f}° "
          f"(物体光の下端 {-deg(s_obj):.2f}° より下なら重ならない)")
    if s_min > s_max + 1e-12:
        print("    !! この CGH ピッチでは物体光・零次光・共役像を分離できません。UPSAMPLE を上げてください。")
    elif not (s_min - 1e-12 <= s_ref <= s_max + 1e-12):
        print("    !! 参照光の角度が範囲外です。物体光の縁が零次光または共役像と重なります。")
    return s_obj, s_nyq, s_ref


# =====================================================================
# 3. 多視点画像群の読み込みと RS 平面波面の計算 (並列読み込み)
# =====================================================================
def build_rs_plane():
    u_RS = np.zeros((N_y, N_x), dtype=np.complex64)

    def load_row(iy):
        # 行ごとに乱数系列を固定 -> スレッドの実行順に関係なく同じ結果になる
        rng = np.random.default_rng([RANDOM_SEED, iy])
        amp = np.zeros((I_views, N_px, M_px), dtype=np.float32)
        miss = 0
        for ix in range(I_views):
            img = cv2.imread(os.path.join(image_folder, f"view_{iy:03d}_{ix:03d}.png"),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                miss += 1
                continue
            if img.shape != (N_px, M_px):
                cy, cx = img.shape[0] // 2, img.shape[1] // 2
                img = img[cy - N_px // 2: cy + N_px // 2, cx - M_px // 2: cx + M_px // 2]
            amp[ix] = img
        amp[amp <= PIXEL_THRESHOLD] = 0.0
        np.sqrt(amp, out=amp)
        random_phase = rng.uniform(0, 2 * np.pi, size=amp.shape).astype(np.float32)
        complex_light = amp * np.exp(1j * random_phase)
        RS_val = sfft.fftshift(sfft.fft2(sfft.ifftshift(complex_light, axes=(1, 2)),
                                         axes=(1, 2)), axes=(1, 2))
        # (ix, r, c) -> (r, ix*M_px + c) に並べて RS 面の 1 行ぶん (高さ N_px) に書き込む
        u_RS[iy * N_px:(iy + 1) * N_px] = RS_val.transpose(1, 0, 2).reshape(N_px, N_x)
        return miss

    print(f"--> Constructing RS plane ({N_x} x {N_y}, {READ_THREADS} threads)...")
    t0 = time.time()
    missing = 0
    with ThreadPoolExecutor(READ_THREADS) as ex:
        for k, miss in enumerate(ex.map(load_row, range(J_views)), 1):
            missing += miss
            if k % 64 == 0 or k == J_views:
                print(f"    {k}/{J_views} rows  ({time.time() - t0:.0f} s)")
    if missing:
        print(f"    !! 見つからなかった画像: {missing} 枚 (0 として扱いました)")
    if missing == I_views * J_views:
        raise FileNotFoundError("画像ファイルが見つかりません。パスを確認してください。")
    print("--> RS plane is created.")
    return u_RS


def rs_brightness_map():
    """summary 用: 各視点の平均輝度 (= RS 面のどこに光があるか)"""
    mean = np.zeros((J_views, I_views), dtype=np.float32)

    def load_row(iy):
        for ix in range(I_views):
            img = cv2.imread(os.path.join(image_folder, f"view_{iy:03d}_{ix:03d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = img.astype(np.float32)
            img[img <= PIXEL_THRESHOLD] = 0.0
            mean[iy, ix] = img.mean()

    print(f"--> Reading multiview images for the summary ({READ_THREADS} threads)...")
    with ThreadPoolExecutor(READ_THREADS) as ex:
        list(ex.map(load_row, range(J_views)))
    return mean


# =====================================================================
# 4. タイル分割 ASM 伝搬 + 0.54um への補間 + オフアクシス 2 値化
# =====================================================================
def asm_transfer(n, z, wl, p):
    """n x n タイル用の伝達関数 (fftshift なしの並び)"""
    f = sfft.fftfreq(n, d=p)
    w2 = (wl * f) ** 2
    sq = 1.0 - w2[:, None] - w2[None, :]
    H = np.exp(1j * (2 * np.pi / wl) * z * np.sqrt(np.maximum(sq, 0.0)))
    H[sq <= 0] = 0.0
    return H.astype(np.complex64)


def crop_padded(arr, y0, y1, x0, x1):
    """arr[y0:y1, x0:x1] を切り出す。範囲外は 0 で埋める"""
    out = np.zeros((y1 - y0, x1 - x0), dtype=arr.dtype)
    sy0, sy1 = max(y0, 0), min(y1, arr.shape[0])
    sx0, sx1 = max(x0, 0), min(x1, arr.shape[1])
    if sy0 < sy1 and sx0 < sx1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def fourier_upsample(U, f):
    """スペクトル U (fftshift なし) を f 倍サイズの配列の低周波側に埋め込む"""
    n0, n1 = U.shape
    h0, h1 = n0 // 2, n1 // 2
    big = np.zeros((n0 * f, n1 * f), dtype=np.complex64)
    big[:h0, :h1] = U[:h0, :h1]
    big[:h0, -h1:] = U[:h0, -h1:]
    big[-h0:, :h1] = U[-h0:, :h1]
    big[-h0:, -h1:] = U[-h0:, -h1:]
    return big


def error_diffuse_rows(a, carry):
    """a (0〜1) を行ごとに左から右へ 1 次の誤差拡散で二値化する (全行を同時に処理)。
    量子化ノイズは横方向の高い空間周波数 (= ±8.79° より外側) に押し出される。
    carry: 左隣のタイルから引き継ぐ誤差 (行ごと)"""
    aT = np.ascontiguousarray(a.T)
    out = np.empty(aT.shape, dtype=bool)
    e = carry
    for j in range(aT.shape[0]):
        v = aT[j] + e
        b = v >= 0.5
        out[j] = b
        e = v - b
    return out.T, e


def rs_rms(u_RS):
    """CGH 面での h = Re(O R*) の標準偏差の見積もり (伝搬はエネルギー保存、補間で 1/UPSAMPLE^2 倍)"""
    p = 0.0
    for s in range(0, u_RS.shape[0], 1024):
        blk = u_RS[s:s + 1024]
        p += float(np.sum(blk.real.astype(np.float64) ** 2 + blk.imag.astype(np.float64) ** 2))
    scale = 1.0 if RS_INTERP == "nearest" else UPSAMPLE ** 2
    return np.sqrt(p / u_RS.size / 2) / scale


def generate_cgh(u_RS, s_obj, s_ref):
    T, M, IM, up = CALC_TILE, CALC_MARGIN, INTERP_MARGIN, UPSAMPLE
    spread = abs(D_val) * np.tan(np.arcsin(s_obj)) / pitch_calc
    assert M >= spread + IM, f"CALC_MARGIN={M} が光の広がり {spread:.0f} px + 補間用 {IM} px より小さい"
    assert N_y % T == 0 and N_x % T == 0
    h_rms = rs_rms(u_RS) if BINARIZE == "error_diffusion" else None

    Ny_c, Nx_c = N_y * up, N_x * up
    k = 2 * np.pi / wavelength
    H = asm_transfer(T + 2 * M, D_val, wavelength, pitch_calc)
    packed = np.zeros((Ny_c, Nx_c // 8), dtype=np.uint8)

    prev_size = min(2048, Ny_c)
    pb = Ny_c // prev_size
    preview = np.zeros((Ny_c // pb, Nx_c // pb), dtype=np.float32)
    ones = 0

    n_ty, n_tx = N_y // T, N_x // T
    print(f"--> Propagating RS -> CGH (z = {D_val*1e3:+.1f} mm) in {n_ty}x{n_tx} tiles "
          f"({T}+2x{M} px @ {pitch_calc*1e6:.2f} um), {RS_INTERP} interpolation to "
          f"{pitch_cgh*1e6:.2f} um, off-axis binarization ({BINARIZE})...")
    t0 = time.time()
    for ty in range(n_ty):
        carry = np.zeros(T * up, dtype=np.float32)
        for tx in range(n_tx):
            y0, x0 = ty * T, tx * T
            u = crop_padded(u_RS, y0 - M, y0 + T + M, x0 - M, x0 + T + M)
            U = sfft.fft2(u, workers=-1, overwrite_x=True)
            del u
            U *= H
            o = sfft.ifft2(U, workers=-1, overwrite_x=True)
            del U

            # 2.16um の物体光 -> 0.54um (RS 面の 1 画素 = CGH の up x up 画素)
            if RS_INTERP == "nearest":
                o = o[M:M + T, M:M + T]
                o = np.repeat(np.repeat(o, up, axis=0), up, axis=1)
            else:
                oc = sfft.fft2(o[M - IM:M + T + IM, M - IM:M + T + IM], workers=-1)
                del o
                big = fourier_upsample(oc, up)
                del oc
                o = sfft.ifft2(big, workers=-1, overwrite_x=True)
                del big
                o = o[IM * up:(IM + T) * up, IM * up:(IM + T) * up]

            # 参照光 R = exp(+i k sinθr y) (下から上向き) との干渉成分 h = Re(O R*) を 2 値化
            rows = y0 * up + np.arange(T * up)
            y_up = (Ny_c / 2 - rows - 0.5) * pitch_cgh
            carrier = np.exp(-1j * np.mod(k * s_ref * y_up, 2 * np.pi)).astype(np.complex64)
            h = o.real * carrier.real[:, None] - o.imag * carrier.imag[:, None]
            del o
            if BINARIZE == "kinoform":
                bits = h >= 0          # cos(arg(O R*)) >= 0 -> bit 1 (位相 0), それ以外 -> bit 0 (位相 π)
            else:
                h *= 0.5 / (ED_SCALE * h_rms)
                h += 0.5
                np.clip(h, 0.0, 1.0, out=h)
                bits, carry = error_diffuse_rows(h, carry)
            del h

            ry, rx = y0 * up, x0 * up
            packed[ry:ry + T * up, rx // 8:(rx + T * up) // 8] = np.packbits(bits, axis=1)
            ones += int(np.count_nonzero(bits))
            preview[ry // pb:(ry + T * up) // pb, rx // pb:(rx + T * up) // pb] = \
                bits.reshape(T * up // pb, pb, T * up // pb, pb).mean(axis=(1, 3))
            del bits
            gc.collect()
        print(f"    tile row {ty + 1}/{n_ty}  ({time.time() - t0:.0f} s)")

    print(f"--> bit 1 の割合: {ones / (Ny_c * Nx_c) * 100:.2f} %")
    return packed, preview


# =====================================================================
# 5. 再生シミュレーション (タイル分割, 目は無限遠)
# =====================================================================
def unpack_tile(packed, y0, y1, x0, x1):
    """packed から [y0:y1, x0:x1] のビットを取り出す (x0, x1 は 8 の倍数)。範囲外は inside=False"""
    Ny_c, Nx_c = packed.shape[0], packed.shape[1] * 8
    bits = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    inside = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    sy0, sy1 = max(y0, 0), min(y1, Ny_c)
    sx0, sx1 = max(x0, 0), min(x1, Nx_c)
    if sy0 < sy1 and sx0 < sx1:
        bits[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
            np.unpackbits(packed[sy0:sy1, sx0 // 8:sx1 // 8], axis=1)
        inside[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = True
    return bits, inside


def transmittance(bits, inside):
    if ENCODING == "amplitude":
        return bits.astype(np.complex64)      # 範囲外は bits = 0 = 遮光
    t = np.where(bits == 1, np.complex64(1.0), np.complex64(np.exp(1j * PHASE_DEPTH)))
    t[~inside] = 0.0                          # CGH の外側は遮光
    return t


def simulate_views(packed, s_ref):
    Ny_c, Nx_c = packed.shape[0], packed.shape[1] * 8
    T, M, ds = SIM_TILE, SIM_MARGIN, SIM_DOWNSAMPLE
    n = T + 2 * M
    assert Ny_c % T == 0 and Nx_c % T == 0 and n % ds == 0 and T % ds == 0 and M % ds == 0
    p_out = pitch_cgh * ds
    ns = n // ds
    fs = sfft.fftfreq(ns, d=p_out)
    fc = np.sin(np.radians(VIEW_APERTURE_DEG / 2)) / wavelength
    assert fc < 0.5 / p_out, "瞳のカットオフが間引き後のナイキストを超えています。SIM_DOWNSAMPLE を下げてください"
    pupil = (np.sqrt(fs[:, None] ** 2 + fs[None, :] ** 2) <= fc).astype(np.complex64)

    # t のスペクトルのどこを切り出すか (FFT のビン番号)。
    # 縦: 照明光 R が下から当たるので、t の行方向の周波数 +sinθr/λ が正面 (0°) に出る。
    # 横: 視点の方向 sinθ/λ。
    kidx = np.rint(sfft.fftfreq(ns, d=1.0 / ns)).astype(np.int64)
    c_y = int(round(s_ref * n * pitch_cgh / wavelength))
    views = []
    for name, th in VIEWS:
        c_x = int(round(np.sin(np.radians(th)) * n * pitch_cgh / wavelength))
        views.append(dict(name=name, theta=th, c_x=c_x,
                          ry=(kidx + c_y) % n, rx=(kidx + c_x) % n,
                          field=np.zeros((Ny_c // ds, Nx_c // ds), dtype=np.complex64)))

    spec = np.zeros((SPEC_BINS, SPEC_BINS), dtype=np.float64)

    n_ty, n_tx = Ny_c // T, Nx_c // T
    print(f"--> Simulating reconstruction ({len(views)} views, {n_ty}x{n_tx} tiles of {T}+2x{M} px, "
          f"output {p_out*1e6:.2f} um/px)...")
    t0 = time.time()
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, x0 = ty * T, tx * T
            bits, inside = unpack_tile(packed, y0 - M, y0 + T + M, x0 - M, x0 + T + M)
            t = transmittance(bits, inside)
            del bits, inside
            G = sfft.fft2(t, workers=-1, overwrite_x=True)
            del t

            # ---- ホログラムの角度スペクトル (全タイルの平均) ----
            P = np.abs(G)
            P *= P
            spec += cv2.resize(sfft.fftshift(P), (SPEC_BINS, SPEC_BINS), interpolation=cv2.INTER_AREA)
            del P

            # ---- 各視点: 窓 (瞳) を切り出して間引いた複素振幅を作る ----
            R0, C0 = y0 - M, x0 - M
            for v in views:
                small = G[np.ix_(v["ry"], v["rx"])]
                small *= pupil
                f = sfft.ifft2(small, workers=-1, overwrite_x=True)
                # タイルの原点の違いによる位相を全体座標にそろえる (後で全体を伝搬するため)
                ph = ((c_y * R0) % n + (v["c_x"] * C0) % n) / n
                f *= np.complex64(np.exp(-2j * np.pi * ph))
                v["field"][y0 // ds:(y0 + T) // ds, x0 // ds:(x0 + T) // ds] = \
                    f[M // ds:(M + T) // ds, M // ds:(M + T) // ds]
                del small, f
            del G
            gc.collect()
        print(f"    tile row {ty + 1}/{n_ty}  ({time.time() - t0:.0f} s)")

    # ---- 虚像の位置まで伝搬してピントを合わせる (視点ごとに FFT 1 回、ピント位置ごとに逆 FFT) ----
    print(f"--> Refocusing views to z = {', '.join(f'{z*1e3:+.0f}' for z in FOCUS_ZS)} mm from the CGH...")
    pad = REFOCUS_PAD
    images = []
    for v in views:
        F = np.pad(v["field"], pad)
        Fq = sfft.fft2(F, workers=-1, overwrite_x=True)
        del F, v["field"]
        fy = sfft.fftfreq(Fq.shape[0], d=p_out)
        fx = sfft.fftfreq(Fq.shape[1], d=p_out) + v["c_x"] / (n * pitch_cgh)   # 視点の方向を足し戻す
        sq = 1.0 / wavelength ** 2 - fx[None, :] ** 2 - fy[:, None] ** 2
        kz = (2 * np.pi * np.sqrt(np.maximum(sq, 0.0))).astype(np.float32)
        evan = sq <= 0
        del sq
        for z in FOCUS_ZS:
            Hz = np.exp(1j * z * kz).astype(np.complex64)
            Hz[evan] = 0.0
            Hz *= Fq
            out = sfft.ifft2(Hz, workers=-1, overwrite_x=True)[pad:-pad, pad:-pad]
            images.append(dict(name=v["name"], theta=v["theta"], focus=z,
                               img=(np.abs(out) ** 2).astype(np.float32)))
            del Hz, out
        del Fq, kz, evan
        gc.collect()
    return images, spec


def far_field_map(spec, s_nyq, s_ref):
    """t のスペクトルを画素の回折次数ぶん周期拡張し、画素開口の sinc^2 を掛け、
    照明光の傾き s_ref だけ縦にずらした「実際に出ていく光の角度分布」"""
    nb = spec.shape[0]
    S = spec[::-1]                         # 行を下向き -> 上向き (sinθ_y 増加) に
    S = np.tile(S, (3, 3))
    s = -3 * s_nyq + (np.arange(3 * nb) + 0.5) * (2 * s_nyq / nb)
    w = np.sinc(s * pitch_cgh / wavelength) ** 2
    S = S * w[:, None] * w[None, :]
    # 零次光 (1 点に集中) で正規化すると他が見えないので、上位 0.01% で正規化して零次光は飽和させる
    S = np.log10(np.maximum(S / np.percentile(S, 99.99), 1e-12))
    extent = [s[0], s[-1], s[0] + s_ref, s[-1] + s_ref]
    return S, extent


# =====================================================================
# 6. 保存とプロット
# =====================================================================
def save_view_images(views, out_dir):
    # 明るさはピント位置ごとに、左・中央・右で共通の基準にそろえる
    for z in FOCUS_ZS:
        group = [v for v in views if v["focus"] == z]
        gmax = max(float(v["img"].max()) for v in group)
        enh_v = max(float(np.percentile(np.sqrt(v["img"][::4, ::4]), 99.9)) for v in group)
        tag = f"focus{z*1e3:+.0f}mm"
        for v in group:
            raw = v["img"] / gmax if gmax > 0 else v["img"]
            cv2.imwrite(os.path.join(out_dir, f"view_{v['name']}_{v['theta']:+.0f}deg_{tag}_raw.png"),
                        cv2.applyColorMap(np.uint8(raw * 255), cv2.COLORMAP_INFERNO))
            enh = np.clip(np.sqrt(v["img"]) / enh_v, 0, 1) if enh_v > 0 else raw
            v["enh"] = enh
            cv2.imwrite(os.path.join(out_dir, f"view_{v['name']}_{v['theta']:+.0f}deg_{tag}_enhanced.png"),
                        cv2.applyColorMap(np.uint8(enh * 255), cv2.COLORMAP_INFERNO))


def make_summary_figure(original, rs_mean, views, spec, s_obj, s_nyq, s_ref, out_dir):
    Ny_c, Nx_c = N_y * UPSAMPLE, N_x * UPSAMPLE
    hx, hy = Nx_c * pitch_cgh / 2 * 1e3, Ny_c * pitch_cgh / 2 * 1e3
    n_rows = 1 + (len(FOCUS_ZS) if views else 0)
    fig, ax = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows), squeeze=False)

    # Blender の正投影で撮った元の物体 (再構成像と同じ座標になるように撮ったもの)
    a = ax[0, 0]
    if original is not None:
        im = a.imshow(original, cmap="gray", vmin=0, vmax=1, extent=[-hx, hx, -hy, hy])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04).set_label("pixel value / 255")
    else:
        a.set_facecolor("black")
        a.set_xlim(-hx, hx)
        a.set_ylim(-hy, hy)
        a.text(0, 0, f"ORIGINAL_IMAGE not found\n{ORIGINAL_IMAGE}", color="white",
               ha="center", va="center", fontsize=9, wrap=True)
    a.set_title(f"Original (Blender orthographic, {N_x*pitch_calc*1e3:.2f} mm)")
    a.set_xlabel("x [mm]")
    a.set_ylabel("y [mm]")

    # RS 面 (= CGH と同じ 70.78 mm 角) のどこに光があるか: 1 画素 = 1 視点の平均輝度
    im = ax[0, 1].imshow(rs_mean, cmap="gray", vmin=0, vmax=max(float(rs_mean.max()), 1e-6),
                         extent=[-hx, hx, -hy, hy])
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046, pad=0.04).set_label("mean pixel value of the view (0-255)")
    ax[0, 1].set_title(f"RS plane: mean brightness of each view\n"
                       f"{I_views}x{J_views} views, {N_x*pitch_calc*1e3:.2f} mm (= CGH area)")
    ax[0, 1].set_xlabel("x [mm]")
    ax[0, 1].set_ylabel("y [mm]")

    if spec is not None:
        S, ext = far_field_map(spec, s_nyq, s_ref)
        a = ax[0, 2]
        im = a.imshow(S, cmap="magma", vmin=-4, vmax=0, extent=ext, origin="lower", aspect="equal")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        cb.set_label("log10 intensity (0 = top 0.01 %)")
        lim = 0.75
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.add_patch(mpatches.Rectangle((-s_obj, -s_obj), 2 * s_obj, 2 * s_obj, fill=False,
                                       ec="cyan", lw=1.2, label="object (±%.2f°)" % deg(s_obj)))
        a.add_patch(mpatches.Rectangle((-s_obj, 2 * s_ref - s_obj), 2 * s_obj, 2 * s_obj, fill=False,
                                       ec="lime", lw=1.2, ls="--", label="conjugate"))
        a.plot([0], [s_ref], "o", mfc="none", mec="white", ms=10, label="zero order (%.2f°)" % deg(s_ref))
        for name, th in VIEWS:
            a.add_patch(mpatches.Circle((np.sin(np.radians(th)), 0.0),
                                        np.sin(np.radians(VIEW_APERTURE_DEG / 2)),
                                        fill=False, ec="yellow", lw=1.0))
        ticks = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
        tpos = [np.sin(np.radians(d)) for d in ticks]
        a.set_xticks(tpos, [f"{d}°" for d in ticks])
        a.set_yticks(tpos, [f"{d}°" for d in ticks])
        a.set_xlabel("horizontal angle")
        a.set_ylabel("vertical angle (up +)")
        a.set_title("Far-field angular spectrum (log10, illuminated by R)\n"
                    "yellow: viewing windows")
        a.legend(loc="lower right", fontsize=8)

    for row, z in enumerate(FOCUS_ZS if views else [], 1):
        for col, v in enumerate([v for v in views if v["focus"] == z][:3]):
            img = cv2.resize(v["enh"], (1024, 1024), interpolation=cv2.INTER_AREA)
            a = ax[row, col]
            im = a.imshow(img, cmap="inferno", vmin=0, vmax=1, extent=[-hx, hx, -hy, hy])
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04).set_label(
                "sqrt(intensity), 1 = top 0.1 % (common to all views)")
            a.set_title(f"{v['name']} ({v['theta']:+.0f}°), eye at ∞, aperture {VIEW_APERTURE_DEG:.1f}°\n"
                        f"focus {z*1e3:+.0f} mm from CGH")
            a.set_xlabel("x [mm]")
            a.set_ylabel("y [mm]")

    plt.tight_layout()
    path = os.path.join(out_dir, "summary.png")
    fig.savefig(path, dpi=110)
    print(f"--> Wrote {path}")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


# =====================================================================
# 7. 計算実行
# =====================================================================
def main():
    global OUTPUT_DIR
    if OUTPUT_DIR is None:
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"OffAxis_Kinoform_Glassshoes216_D{D_val*1e3:+.0f}mm_"
                                  f"{pitch_cgh*1e6:.2f}um".replace(".", "p"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--> RS plane: {N_x} x {N_y} @ {pitch_calc*1e6:.2f} um "
          f"({N_x*pitch_calc*1e3:.2f} mm x {N_y*pitch_calc*1e3:.2f} mm)")
    print(f"--> CGH     : {N_x*UPSAMPLE} x {N_y*UPSAMPLE} @ {pitch_cgh*1e6:.2f} um, {BINARIZE}, {ENCODING}")
    print(f"--> Object -> RS {z_rs*1e3:.0f} mm, RS -> CGH {D_val*1e3:+.0f} mm "
          f"(virtual image {-(z_rs + D_val)*1e3:.0f} mm behind the CGH)")
    print(f"--> Output  : {OUTPUT_DIR}")

    s_obj, s_nyq, s_ref = reference_geometry()

    cgh_path = os.path.join(OUTPUT_DIR, "cgh_bits_packed.npy")
    prev_path = os.path.join(OUTPUT_DIR, "cgh_preview_blockmean.npy")
    meta_path = os.path.join(OUTPUT_DIR, "cgh_meta.json")
    saved_threshold = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            saved_threshold = json.load(f).get("pixel_threshold")
    reusable = REUSE_CGH and os.path.exists(cgh_path) and os.path.exists(prev_path)
    if reusable and saved_threshold != PIXEL_THRESHOLD:
        print(f"--> Saved CGH used pixel_threshold={saved_threshold}, now {PIXEL_THRESHOLD}: regenerating.")
        reusable = False
    if reusable:
        # 多視点画像を撮り直していたら作り直す
        newest = max((e.stat().st_mtime for e in os.scandir(image_folder) if e.name.startswith("view_")),
                     default=0.0)
        if newest > os.path.getmtime(cgh_path):
            print("--> Multiview images are newer than the saved CGH: regenerating.")
            reusable = False
    if reusable:
        print(f"--> Reusing saved CGH: {cgh_path}")
        packed = np.load(cgh_path)
        preview = np.load(prev_path)
    else:
        u_RS = build_rs_plane()
        packed, preview = generate_cgh(u_RS, s_obj, s_ref)
        del u_RS
        gc.collect()

        np.save(cgh_path, packed)
        np.save(prev_path, preview)
        meta = dict(
            generated=datetime.datetime.now().isoformat(timespec="seconds"),
            image_folder=image_folder,
            shape_rows_cols=[packed.shape[0], packed.shape[1] * 8],
            pixel_pitch_m=pitch_cgh, calc_pitch_m=pitch_calc, wavelength_m=wavelength,
            rs_interp=RS_INTERP, pixel_threshold=PIXEL_THRESHOLD,
            object_to_rs_m=z_rs, rs_to_cgh_m=D_val, virtual_image_depth_m=-(z_rs + D_val),
            reference_angle_deg=float(deg(s_ref)),
            reference_direction="from below, travelling upward (vertical tilt only)",
            binarize=BINARIZE, encoding=ENCODING,
            bit_meaning=("bit 1 = phase 0, bit 0 = phase pi" if ENCODING == "phase"
                         else "bit 1 = transmit, bit 0 = block"),
            storage="np.packbits(axis=1), MSB = left pixel, row 0 = top",
            random_seed=RANDOM_SEED,
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"--> Saved CGH: {cgh_path} ({packed.nbytes/2**20:.0f} MiB)")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "cgh_preview_blockmean.png"), np.uint8(preview * 255))
    cc = min(1024, packed.shape[0])
    r0, c0 = packed.shape[0] // 2 - cc // 2, packed.shape[1] * 4 - cc // 2
    center_crop = np.unpackbits(packed[r0:r0 + cc, c0 // 8:(c0 + cc) // 8], axis=1).astype(np.float32)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "cgh_center_crop.png"), np.uint8(center_crop * 255))

    views, spec = [], None
    if RUN_SIMULATION:
        views, spec = simulate_views(packed, s_ref)
        np.save(os.path.join(OUTPUT_DIR, "far_field_spectrum.npy"), spec)
        save_view_images(views, OUTPUT_DIR)

    original = None
    if ORIGINAL_IMAGE and os.path.exists(ORIGINAL_IMAGE):
        original = cv2.imread(ORIGINAL_IMAGE, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    rs_mean = rs_brightness_map()
    make_summary_figure(original, rs_mean, views, spec, s_obj, s_nyq, s_ref, OUTPUT_DIR)
    print("--> Done.")


if __name__ == "__main__":
    main()
