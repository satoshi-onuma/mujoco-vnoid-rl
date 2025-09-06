import sys
import os

# ビルドして生成された .so ファイルのあるディレクトリをPythonのパスに追加
# これにより、vnoid_rl_envをインポートできるようになる
build_path = os.path.abspath("./build/controller/vnoid_rl_env")
sys.path.append(build_path)

print(f"モジュールを探すパス: {build_path}")

try:
    # ★ C++で作成したモジュールをインポート
    import vnoid_rl_env
    print("✅ モジュールのインポートに成功！")
except ImportError as e:
    print(f"❌ モジュールのインポートに失敗しました: {e}")
    sys.exit(1)

# モデルファイルのパスを正しく指定してください
model_xml_path = "model/sample_robot/sample_robot_mujoco.xml"
# model_xml_path = "share/model/sample_robot/sample_robot_mujoco.xml" # もしルートから実行するならこちら

print("\nC++環境のインスタンスを作成します...")
# C++側のVnoidEnvクラスのインスタンスを作成
env = vnoid_rl_env.VnoidEnv(model_xml_path)
print("✅ インスタンス化に成功！")

print("\n環境をリセットします...")
obs = env.reset()
print(f"リセット後の観測 (obs): {obs}")

print("\n環境を1ステップ進めます...")
# ダミーのアクション（着地点オフセットx=0.01, y=-0.01）を渡す
dummy_action = [0.01, -0.01] 
obs, reward, terminated, info = env.step(dummy_action)
print(f"ステップ後の観測 (obs): {obs}")
print(f"報酬 (reward): {reward}")
print(f"終了フラグ (terminated): {terminated}")
print(f"情報 (info): {info}")

print("\n✅ テスト完了！")
