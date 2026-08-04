import bpy
import numpy as np
import os

# =====================================================================
# 1. パラメータ設定 (光学理論値および幾何条件の統一)
# =====================================================================
grid_x = 256   # X軸方向の視点数
grid_y = 256   # Z軸方向（高さ方向）の視点数
num_views_total = grid_x * grid_y  # 合計 65536 枚

resolution_x = 32  # 投影画像の横幅 [px]
resolution_y = 32  # 投影画像の高さ [px]

sphere_radius = 0.005  # 球の半径 5mm (直径 10mm = 0.01m)

# カメラ位置パラメータ
distance = 0.015       # 原点からカメラまでの距離 (15mm -> Y = -0.015m)

# カメラから球の最前面までの実際の物理距離 (15mm - 5mm = 10mm)
dist_to_front = distance - sphere_radius

# 光学パラメータ (Python側と一致させる)
wavelength_cg = 633e-9
pitch_cg = 2.0e-6

# RS平面上の1ブロックの物理サイズ (32 px * 2.0 μm = 64 μm = 0.000064 m)
block_size = resolution_x * pitch_cg

# シフト幅の計算
max_shift = ((grid_x - 1) * block_size) / 2.0

# 平行投影スケール (指定の 0.01 m = 10 mm)
ortho_scale_value = 0.01

# 画像の保存先フォルダ
output_dir = f"C:/Lab/Orthographic_Sphere_multiview_output_fullparallax_{grid_x}x{grid_y}/"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 2. カメラのセットアップ (平行投影: Orthographic)
# =====================================================================
if "Camera" in bpy.data.objects:
    camera = bpy.data.objects["Camera"]
else:
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object

bpy.context.scene.camera = camera

# 既存のコンストレイント（Track To 等）の解除
for constraint in camera.constraints:
    camera.constraints.remove(constraint)

# カメラの回転角 (RX=90°, RY=0°, RZ=0° で Y軸正方向＝原点を直角に向く)
camera.rotation_euler = (np.radians(90.0), 0.0, 0.0)

# --- 【平行投影 (Orthographic) の詳細設定】 ---
camera.data.type = 'ORTHO'

# 平行投影スケール (撮影枠の物理サイズ = 10 mm)
camera.data.ortho_scale = ortho_scale_value

# 描画開始・終了範囲 (Clipping)
# 球最前面(10mm手前)のくり抜きを防ぐため 0.0001m (0.1mm) から描画開始
camera.data.clip_start = 0.0001
camera.data.clip_end = 100.0

# レンダリング設定
scene = bpy.context.scene
scene.render.resolution_x = resolution_x
scene.render.resolution_y = resolution_y
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'BW'  # 8bit グレースケール

# =====================================================================
# 3. 2D平面軌道 (X-Z平面) での Full-Parallax 撮影処理
# =====================================================================
x_positions = np.linspace(-max_shift, max_shift, grid_x)
z_positions = np.linspace(-max_shift, max_shift, grid_y)

view_count = 0

for iy, cam_z in enumerate(z_positions):
    for ix, cam_x in enumerate(x_positions):
        view_count += 1
        
        # カメラ位置: 原点(0,0,0)にある球体に向けて Y = -0.015m から撮影
        camera.location = (cam_x, -distance, cam_z)
        bpy.context.view_layer.update()
        
        file_path = os.path.join(output_dir, f"view_{iy:03d}_{ix:03d}.png")
        scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

print("--> [平行投影] Full-Parallax 撮影が正常に完了しました！")