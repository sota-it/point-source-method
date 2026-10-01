import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import grcwa

# Windows のコンソール (cp932) で表示できない文字があっても停止しないようにする
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# =====================================================================
# quadrumer メタ原子の回折効率スペクトル (論文 Fig. 3c, d と同じ形式)
#   参考: Jang et al., Adv. Opt. Mater. 9, 2100678 (2021), Fig. 3c, d
#
#   各波長 λ で、入射角を θ(λ) = ±sin^-1(λ / (n_sub P)) (基板内) に合わせる。
#   この角度では1次回折光がちょうど垂直方向 (法線方向) に出るので、
#   下の横軸 (波長) と上の横軸 (入射角) は 1対1 に対応する。
#   縦軸 T0 = 垂直方向に出る回折次数の透過効率 (+θ: (-1, 0) 次, -θ: (+1, 0) 次)。
#
#   RCWA ライブラリ: grcwa (pip install grcwa)
# =====================================================================
grcwa.set_backend('numpy')

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RCWA_output")
os.makedirs(out_dir, exist_ok=True)

# =====================================================================
# 1. パラメータ
# =====================================================================
P = 700e-9                   # 周期
n_sub = 1.45                 # 基板 (SiO2) の屈折率 (分散は無視)
lam_design = 633e-9          # 設計波長 (図の点線)

# quadrumer (論文 Fig. 3a)。配置: (-,+) s1, (-,-) s2, (+,+) s3, (+,-) s4、隣接ピラーの中心間距離 g
quad = dict(s1=100e-9, s2=125e-9, s3=150e-9, s4=100e-9, g=175e-9, t=350e-9)
# True: quadrumer を 180° 回転 (= s2 と s3 の位置を入れ替え)。論文 Fig. 3c (+θ で強く回折) と同じ向きになる。
#   論文の図には各ピラーの大きさと位置の対応が明記されていないため、配置 (-,-) s2, (+,+) s3 のままでは
#   -θ 側で強く回折する向きになっていた。180° 回転は論文のメタ原子1 ↔ メタ原子2 の関係と同じ。
rotate180 = True

# ピラー材料の屈折率。
#   n_pillar_csv に「波長[nm], n, k」の CSV (エリプソメトリの測定値など) を指定すると波長分散を考慮する。
#   None の場合は一定値 n_pillar_const を全波長で使う (論文の a-Si の 633 nm での値)。
#   ※ a-Si は短波長ほど屈折率・吸収が大きくなるため、一定値では 550 nm 以下の結果は定量的でない。
n_pillar_csv = None
n_pillar_const = 3.19 + 0.02j

# 計算する波長 (5 nm 刻み + 設計波長)。λ = P (700 nm) ちょうどでは (-2, 0) 次が空気中で真横に出る
# (レイリー異常) ため RCWA が発散するので、上端は 699 nm にする
wavelengths = np.unique(np.concatenate([np.arange(400, 700, 5), [lam_design * 1e9, 699]])) * 1e-9
nG = 401                     # フーリエ次数 (論文条件で 601 との差は 1 % 程度)
Nxy = 200                    # 1周期あたりの誘電率グリッド数 (3.5 nm)


def pillar_index(lam):
    if n_pillar_csv is None:
        return n_pillar_const
    d = np.loadtxt(n_pillar_csv, delimiter=',', comments='#')
    return np.interp(lam * 1e9, d[:, 0], d[:, 1]) + 1j * np.interp(lam * 1e9, d[:, 0], d[:, 2])


def quadrumer_eps(eps_pillar):
    x = (np.arange(Nxy) + 0.5) / Nxy * P - P / 2
    X, Y = np.meshgrid(x, x, indexing='ij')
    h = quad['g'] / 2
    centers = [(-h, +h), (-h, -h), (+h, +h), (+h, -h)]
    sizes = [quad['s1'], quad['s2'], quad['s3'], quad['s4']]
    if rotate180:
        sizes = [quad['s4'], quad['s3'], quad['s2'], quad['s1']]
    eps = np.ones((Nxy, Nxy), dtype=complex)
    for (cx, cy), s in zip(centers, sizes):
        eps[(np.abs(X - cx) < s / 2) & (np.abs(Y - cy) < s / 2)] = eps_pillar
    return eps


def normal_order_efficiency(lam, sign):
    """基板側から θ = sign * sin^-1(λ/(n_sub P)) で TM 入射したときの、垂直方向の回折次数の透過効率"""
    th = sign * np.arcsin(lam / (n_sub * P))
    o = grcwa.obj(nG, [P, 0], [0, P], 1 / lam, th, 0.0, verbose=0)
    o.Add_LayerUniform(0, n_sub ** 2)
    o.Add_LayerGrid(quad['t'], Nxy, Nxy)
    o.Add_LayerUniform(0, 1.0)
    o.Init_Setup()
    o.GridLayer_geteps(quadrumer_eps(pillar_index(lam) ** 2).flatten())
    o.MakeExcitationPlanewave(1, 0, 0, 0, order=0)        # p 偏光 (= TM)
    R, T = o.RT_Solve(normalize=1, byorder=1)
    target = (-sign, 0)                                    # +θ → (-1, 0), -θ → (+1, 0)
    for g, Ti in zip(o.G, T):
        if (int(g[0]), int(g[1])) == target:
            return float(np.real(Ti))
    return 0.0


# =====================================================================
# 2. 波長掃引
# =====================================================================
T_plus = np.zeros(len(wavelengths))
T_minus = np.zeros(len(wavelengths))
t0 = time.time()
for i, lam in enumerate(wavelengths):
    T_plus[i] = normal_order_efficiency(lam, +1)
    T_minus[i] = normal_order_efficiency(lam, -1)
    print(f"--> {lam * 1e9:5.0f} nm (θ = ±{np.degrees(np.arcsin(lam / (n_sub * P))):4.1f} deg): "
          f"T0(+θ) = {T_plus[i] * 100:6.2f} %, T0(-θ) = {T_minus[i] * 100:6.2f} %   [{time.time() - t0:.0f} s]")
    sys.stdout.flush()

np.savetxt(os.path.join(out_dir, "rcwa_spectrum.csv"),
           np.column_stack([wavelengths * 1e9, np.degrees(np.arcsin(wavelengths / (n_sub * P))), T_plus, T_minus]),
           delimiter=',', header="wavelength_nm,theta_sub_deg,T0_plus,T0_minus", comments='')

i_d = np.argmin(np.abs(wavelengths - lam_design))
print(f"\n--> At {wavelengths[i_d] * 1e9:.1f} nm: T0(+θ) = {T_plus[i_d] * 100:.2f} %, T0(-θ) = {T_minus[i_d] * 100:.2f} %")

# =====================================================================
# 3. 描画 (論文 Fig. 3c, d と同じ形式: 下軸 = 波長, 上軸 = 入射角)
# =====================================================================
def lam2ang(l_nm):
    return np.degrees(np.arcsin(np.clip(l_nm * 1e-9 / (n_sub * P), -1, 1)))


def ang2lam(a):
    return np.sin(np.radians(a)) * n_sub * P * 1e9


fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a_, T, sign, color in [(ax[0], T_plus, +1, '#1f3b73'), (ax[1], T_minus, -1, '#d2691e')]:
    a_.plot(wavelengths * 1e9, T, color=color, lw=2)
    a_.axvline(lam_design * 1e9, color='gray', ls='--')
    a_.set_xlim(400, 700); a_.set_ylim(0, 1)
    a_.set_xlabel("Wavelength [nm]"); a_.set_ylabel("$T_0$")
    a_.set_yticks([0, 0.25, 0.5, 0.75, 1]); a_.grid(color='white'); a_.set_facecolor('#f1f3f6')
    top = a_.secondary_xaxis('top', functions=((lambda l, s=sign: s * lam2ang(l)), (lambda a, s=sign: ang2lam(s * a))))
    # 論文と同じく、波長 400, 500, 600, 700 nm の位置に対応する入射角を目盛りにする
    top.set_xticks([sign * lam2ang(l) for l in (400, 500, 600, 700)])
    top.set_xticklabels([f"{sign * lam2ang(l):.0f}" for l in (400, 500, 600, 700)])
    top.set_xlabel("Incident angle [°]")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "rcwa_spectrum.png"), dpi=200)
print(f"--> Saved to '{out_dir}'")
if os.environ.get('NO_SHOW') is None:
    plt.show()
