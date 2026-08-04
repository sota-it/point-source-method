import bpy
import numpy as np
import os

# =====================================================================
# 1. パラメータ設定 (光学理論値との完全一致化)
# =====================================================================
grid_x = 256  # X軸方向の視点数
grid_y = 256  # Z軸方向（高さ方向）の視点数
num_views_total = grid_x * grid_y  # 合計 16384 枚

resolution_x = 32  # 投影画像の横幅 [px]
resolution_y = 32  # 投影画像の高さ [px]

distance = 0.005   # RS平面/原点(0,0,0)からのカメラ距離 [m] (4mm)

# 光学パラメータ (Python側と一致させる)
wavelength_cg = 633e-9
pitch_cg = 2.0e-6

# RS平面上の1ブロックの物理サイズ (64μm)
block_size = resolution_x * pitch_cg

# 127区間分のシフト幅の半分を計算
max_shift = ( (grid_x - 1) * block_size ) / 2.0

# 画像の保存先フォルダ
output_dir = f"C:/Lab/Sphere_multiview_output_fullparallax_{grid_x}x{grid_y}/"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 2. カメラのセットアップ (画角と位置)
# =====================================================================
if "Camera" in bpy.data.objects:
    camera = bpy.data.objects["Camera"]
else:
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object

bpy.context.scene.camera = camera

for constraint in camera.constraints:
    camera.constraints.remove(constraint)

# カメラの回転角 (RX=90°, RY=0°, RZ=0° で Y軸正方向を向く)
camera.rotation_euler = (np.radians(90.0), 0.0, 0.0)

# --- 【重要】カメラの画角をホログラムの最大回折角に完全一致させる ---
theta_max = np.arcsin(wavelength_cg / (2.0 * pitch_cg))
camera.data.angle = 2.0 * theta_max  # Full FOVを設定

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

print(f"--> Full-Parallax 撮影を開始します...")
print(f"--> 視点構成: {grid_x} x {grid_y} (計 {num_views_total} 枚)")
print(f"--> 最大シフト幅: {max_shift*1000:.3f} mm")
print(f"--> カメラ画角: {np.degrees(camera.data.angle):.2f} 度")

view_count = 0

for iy, cam_z in enumerate(z_positions):
    for ix, cam_x in enumerate(x_positions):
        view_count += 1
        
        # カメラ位置: 原点(0,0,0)にある物体に向けて -Y 側から撮影
        camera.location = (cam_x, -distance, cam_z)
        bpy.context.view_layer.update()
        
        file_path = os.path.join(output_dir, f"view_{iy:03d}_{ix:03d}.png")
        scene.render.filepath = file_path
        bpy.ops.render.render(write_still=True)

print("--> Full-Parallax 撮影が正常に完了しました！")