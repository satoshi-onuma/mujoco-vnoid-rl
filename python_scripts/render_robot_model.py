#!/usr/bin/env python3
"""
ロボットモデルを4方向から撮影して個別画像を作成
(front, top, side, perspective)
Usage: python render_robot_model.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from PIL import Image

# モデルパス（python_scripts/から見た相対パス）
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 "../model/sample_robot/sample_robot_mujoco.xml")
)

def render_view(model, data, azimuth, elevation, distance=2.5, lookat=None):
    """指定した角度からロボットを撮影"""
    # レンダラー作成
    renderer = mujoco.Renderer(model, height=800, width=800)
    
    # ロボットを初期姿勢に
    mujoco.mj_resetData(model, data)

    # 腕を横に上げる（追加）
    # 右腕のロール関節
    #r_upperarm_r_id = model.joint('R_UPPERARM_R').id
    data.qpos[19] = 90 * np.pi / 180  # -90度（ラジアン）
    
    # 左腕のロール関節
    #l_upperarm_r_id = model.joint('L_UPPERARM_R').id
    data.qpos[12] = -90 * np.pi / 180   # 90度（ラジアン）
    
    mujoco.mj_forward(model, data)
    
    # カメラ設定
    camera = mujoco.MjvCamera()
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    
    if lookat is None:
        # ロボットの重心を中心に
        lookat = np.array([0.0, 0.0, 0.9])  # 初期高さ0.9m
    camera.lookat[:] = lookat
    
    # レンダリングオプション
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    
    # 背景を白に
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    
    # レンダリング実行
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    img = renderer.render()
    
    return img


def create_robot_figure(model, data, output_dir='robot_renders'):
    """4つの視点から撮影して個別に保存"""
    
    print("🎨 ロボットモデルをレンダリング中...")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 4方向から撮影
    views = {
        'front': {'params': {'azimuth': 90, 'elevation': 0}, 'title': 'Front View'},        # 正面
        'top': {'params': {'azimuth': 90, 'elevation': -89}, 'title': 'Top View'},         # 上面
        'side': {'params': {'azimuth': 0, 'elevation': 0}, 'title': 'Side View'},          # 側面
        'perspective': {'params': {'azimuth': 45, 'elevation': -20}, 'title': 'Perspective View'}  # 透視図
    }
    
    saved_files = []
    
    for view_name, view_info in views.items():
        print(f"  📸 {view_info['title']}をレンダリング中...")
        
        # レンダリング
        img = render_view(model, data, **view_info['params'])
        
        # PILで保存
        output_path = os.path.join(output_dir, f'{view_name}.png')
        img_pil = Image.fromarray(img)
        img_pil.save(output_path, dpi=(300, 300))
        
        saved_files.append(output_path)
        print(f"    ✅ 保存完了: {output_path}")
    
    print(f"\n✅ 全レンダリング完了（{len(saved_files)}ファイル）")
    return saved_files


def main():
    print("=" * 70)
    print("🤖 ロボットモデル 4方向レンダリング")
    print("=" * 70)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ エラー: モデルファイルが見つかりません")
        print(f"パス: {MODEL_PATH}")
        return
    
    print(f"📂 モデルファイル: {MODEL_PATH}")
    
    # モデルロード
    print("⏳ モデルをロード中...")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        print("✅ モデルロード完了")
    except Exception as e:
        print(f"❌ モデルロード失敗: {e}")
        return
    
    # 図を作成
    output_dir = "robot_renders"
    saved_files = create_robot_figure(model, data, output_dir)
    
    print("\n" + "=" * 70)
    print("🎉 完了！")
    print(f"📁 出力ディレクトリ: {output_dir}/")
    print(f"📄 出力ファイル数: {len(saved_files)}")
    for f in saved_files:
        print(f"   - {f}")
    print("=" * 70)


if __name__ == "__main__":
    main()