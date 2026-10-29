"""
Glassshoes216 Full-Parallax 多視点撮影スクリプト (並列化対応)

  - 視点数   : 512 x 512 (計 262144 枚)
  - 解像度   : 64 x 64 px
  - 波長     : 660 nm
  - 画素ピッチ: 2.16 um
  - 出力先   : C:/Lab/Glassshoes216_64px_multiview_output_fullparallax_512x512_z0005/

このファイルは 2 通りに使えます。

[A] 並列実行 (推奨) : 通常の Python から起動すると、Blender をバックグラウンドで
    N 個同時に立ち上げ、視点を N 分割して撮影します。
        python Glassshoes216_multiview_render.py ^
            --blender "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe" ^
            --blend   "C:/Lab/Glassshoes.blend" ^
            --workers 8

[B] 単体実行 : Blender の Text Editor で開いて実行するか、
        blender -b Glassshoes.blend --python Glassshoes216_multiview_render.py
    とすると 1 プロセスで全視点を撮影します。

既に存在する画像はスキップするので、途中で止めても同じコマンドで再開できます。
"""

import os
import sys
import time
import argparse

import numpy as np

# =====================================================================
# 1. パラメータ設定 (光学理論値との完全一致化)
# =====================================================================

grid_x = 512  # X軸方向の視点数
grid_y = 512  # Z軸方向（高さ方向）の視点数

num_views_total = grid_x * grid_y  # 合計 262144 枚

resolution_x = 64  # 投影画像の横幅 [px]
resolution_y = 64  # 投影画像の高さ [px]

distance = 0.005   # RS平面/原点(0,0,0)からのカメラ距離 [m] (5mm)

clip_start = 1e-5  # カメラの手前クリップ [m] (0.01 mm)

# 光学パラメータ (Python側と一致させる)
wavelength_cg = 660e-9

pitch_cg = 2.16e-6

# RS平面上の1ブロックの物理サイズ (138.24μm)
block_size = resolution_x * pitch_cg

# 512視点の場合、全体の幅は block_size * 512。その半分のシフト幅を計算
max_shift = (block_size * grid_x) / 2.0  # (約35.39 mm)

# 画像の保存先フォルダ
output_dir = f"C:/Lab/Glassshoes216_64px_multiview_output_fullparallax_{grid_x}x{grid_y}_z0005/"

# カメラ位置 (元スクリプトと同じ並び)
x_positions = np.linspace(-max_shift + block_size/2.0, max_shift - block_size/2.0, grid_x)
z_positions = np.linspace(max_shift - block_size/2.0, -max_shift + block_size/2.0, grid_y)


def view_path(iy, ix):
    return os.path.join(output_dir, f"view_{iy:03d}_{ix:03d}.png")


# =====================================================================
# 2. ワーカー (Blender 内で実行される撮影処理)
# =====================================================================

def run_worker(worker_id, num_workers, skip_existing=True, samples=None):
    import bpy

    os.makedirs(output_dir, exist_ok=True)

    # --- カメラのセットアップ ---
    # シーンのアクティブカメラを使う (日本語 UI で作ったファイルでは名前が "カメラ" になるため、
    # 名前で探すと見つからず新規カメラ (Clip Start 100 mm) が作られて何も写らなくなる)
    camera = bpy.context.scene.camera
    if camera is None and "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    if camera is None:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
    bpy.context.scene.camera = camera

    # 比較用の正投影で撮ったあとに保存されていても、多視点撮影は必ず透視投影で行う
    camera.data.type = 'PERSP'
    # 物体はカメラから数 mm〜数十 mm にあるので、手前のクリップを十分小さくする
    camera.data.clip_start = clip_start

    for constraint in list(camera.constraints):
        camera.constraints.remove(constraint)

    # カメラの回転角 (RX=90°, RY=0°, RZ=0° で Y軸正方向を向く)
    camera.rotation_euler = (np.radians(90.0), 0.0, 0.0)

    # --- 【重要】カメラの画角をホログラムの最大回折角に完全一致させる ---
    theta_max = np.arcsin(wavelength_cg / (2.0 * pitch_cg))
    camera.data.angle = 2.0 * theta_max  # Full FOVを設定

    # --- レンダリング設定 ---
    scene = bpy.context.scene
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'BW'  # 8bit グレースケール
    # ディザを切る (既定 1.0 だと真っ黒な視点にも画素値 1 の点が散り、CGH でノイズ光になる)
    scene.render.dither_intensity = 0.0

    # --- 高速化設定 ---
    # レンダ間でシーンデータを保持し、毎回の再構築を省く (Cycles で有効)
    scene.render.use_persistent_data = True
    # 複数プロセス同時実行時は CPU スレッドを分け合う (CPU レンダ時の取り合い防止)
    if num_workers > 1:
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = max(1, (os.cpu_count() or 1) // num_workers)

    my_indices = range(worker_id, num_views_total, num_workers)
    n_mine = len(my_indices)

    print(f"--> [worker {worker_id}/{num_workers}] Full-Parallax 撮影を開始します...")
    print(f"--> 視点構成: {grid_x} x {grid_y} (計 {num_views_total} 枚, 担当 {n_mine} 枚)")
    print(f"--> 最大シフト幅: {max_shift*1000:.3f} mm")
    print(f"--> カメラ画角: {np.degrees(camera.data.angle):.2f} 度")
    print(f"--> カメラ: {camera.name}  (clip {camera.data.clip_start*1e3:.3f} mm - {camera.data.clip_end:.1f} m)")
    if scene.render.engine == 'CYCLES':
        if samples is not None:
            scene.cycles.samples = samples
        print(f"--> Cycles samples: {scene.cycles.samples}")
    sys.stdout.flush()

    t0 = time.time()
    rendered = 0
    for n, idx in enumerate(my_indices, 1):
        iy, ix = divmod(idx, grid_x)
        file_path = view_path(iy, ix)
        if skip_existing and os.path.exists(file_path):
            continue

        # カメラ位置: 原点(0,0,0)にある物体に向けて -Y 側から撮影
        camera.location = (float(x_positions[ix]), -distance, float(z_positions[iy]))

        bpy.ops.render.render(write_still=False)

        # 一時ファイルに書いてからリネーム (中断時に壊れた PNG が残らないように)
        tmp_path = file_path + ".tmp.png"
        bpy.data.images["Render Result"].save_render(filepath=tmp_path, scene=scene)
        os.replace(tmp_path, file_path)
        rendered += 1

        if rendered % 500 == 0:
            el = time.time() - t0
            print(f"[worker {worker_id}] {n}/{n_mine}  ({rendered / el:.1f} 枚/s)")
            sys.stdout.flush()

    print(f"--> [worker {worker_id}] 撮影が正常に完了しました！ ({rendered} 枚, {time.time() - t0:.0f} s)")
    sys.stdout.flush()


# =====================================================================
# 3. ランチャー (通常の Python から Blender を N 並列で起動)
# =====================================================================

def run_launcher(argv):
    import subprocess

    p = argparse.ArgumentParser(description="Blender 多視点撮影の並列ランチャー")
    p.add_argument("--blender", required=True, help="blender.exe のパス")
    p.add_argument("--blend", required=True, help="撮影するシーンの .blend ファイル")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="同時に起動する Blender プロセス数")
    p.add_argument("--no-skip", action="store_true", help="既存画像も撮り直す")
    p.add_argument("--samples", type=int, default=None,
                   help="Cycles のサンプル数を上書き (省略時は .blend の設定のまま)")
    args = p.parse_args(argv)

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "_logs")
    os.makedirs(log_dir, exist_ok=True)

    script = os.path.abspath(__file__)
    procs, logs = [], []
    for k in range(args.workers):
        cmd = [args.blender, "-b", args.blend, "--python", script, "--",
               "--worker-id", str(k), "--num-workers", str(args.workers)]
        if args.no_skip:
            cmd.append("--no-skip")
        if args.samples is not None:
            cmd += ["--samples", str(args.samples)]
        log = open(os.path.join(log_dir, f"worker_{k:02d}.log"), "w", encoding="utf-8")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        logs.append(log)

    print(f"--> {args.workers} 個の Blender を起動しました。ログ: {log_dir}")
    t0 = time.time()
    try:
        while any(pr.poll() is None for pr in procs):
            time.sleep(60)
            done = sum(1 for e in os.scandir(output_dir) if e.name.endswith(".png") and not e.name.endswith(".tmp.png"))
            el = time.time() - t0
            print(f"    {done}/{num_views_total} 枚 ({100*done/num_views_total:.1f}%), 経過 {el/60:.1f} 分")
    except KeyboardInterrupt:
        print("--> 中断します。Blender プロセスを終了しています...")
        for pr in procs:
            pr.terminate()
    finally:
        for pr in procs:
            pr.wait()
        for log in logs:
            log.close()

    codes = [pr.returncode for pr in procs]
    if all(c == 0 for c in codes):
        print("--> Full-Parallax 撮影が正常に完了しました！")
    else:
        print(f"--> 異常終了したワーカーがあります (終了コード: {codes})。_logs を確認してください。")


# =====================================================================
# 4. エントリーポイント
# =====================================================================

if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
        in_blender = True
    except ImportError:
        in_blender = False

    if in_blender:
        # Blender 内: "--" 以降の引数を読む (無ければ 1 プロセスで全視点)
        script_args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        wp = argparse.ArgumentParser()
        wp.add_argument("--worker-id", type=int, default=0)
        wp.add_argument("--num-workers", type=int, default=1)
        wp.add_argument("--no-skip", action="store_true")
        wp.add_argument("--samples", type=int, default=None)
        wargs = wp.parse_args(script_args)
        run_worker(wargs.worker_id, wargs.num_workers, skip_existing=not wargs.no_skip,
                   samples=wargs.samples)
    else:
        run_launcher(sys.argv[1:])
