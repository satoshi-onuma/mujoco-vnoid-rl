# ★★★ ニューラルネットワーク設定確認スクリプト ★★★

import os
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from pprint import pprint
from my_humanoid_env import HumanoidVnoidEnv

print("=" * 70)
print("🔍 Vnoid Humanoid ニューラルネットワーク設定確認")
print("=" * 70)

# チェックポイント確認
checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint_id3")
if not os.path.exists(checkpoint_dir):
    print(f"\n❌ エラー: チェックポイントが見つかりません")
    print(f"パス: {checkpoint_dir}")
    print("\n利用可能なチェックポイント:")
    possible_dirs = [
        "./humanoid_vnoid_checkpoint",
        "./humanoid_vnoid_checkpoint_id1",
        "./humanoid_vnoid_checkpoint_id2",
        "./humanoid_vnoid_checkpoint_id3",
        "./humanoid_vnoid_checkpoint_id6",
    ]
    for d in possible_dirs:
        if os.path.exists(d):
            print(f"  - {d}")
    exit(1)

print(f"\n📂 チェックポイント: {checkpoint_dir}")

# Ray初期化
ray.init(logging_level="ERROR")

# チェックポイントからアルゴリズムをロード
print("\n📥 チェックポイントをロード中...")
try:
    algo = Algorithm.from_checkpoint(checkpoint_dir)
    print("✅ ロード完了\n")
except Exception as e:
    print(f"❌ ロード失敗: {e}")
    ray.shutdown()
    exit(1)

# モデル設定を取得
print("=" * 70)
print("📊 モデル設定")
print("=" * 70)

model_config = algo.config.model

# 基本モデル設定
print("\n【基本設定】")
print(f"  フレームワーク: {algo.config.framework}")
print(f"  モデルタイプ: {model_config.get('_model_type', 'default')}")

# 観測空間と行動空間（環境から直接取得）
print("\n【観測・行動空間】")
try:
    env = HumanoidVnoidEnv(enable_rendering=False)
    obs_space = env.observation_space
    action_space = env.action_space
    env.close()
    print(f"  観測空間: {obs_space}")
    print(f"  観測次元: {obs_space.shape if hasattr(obs_space, 'shape') else 'N/A'}")
    print(f"  行動空間: {action_space}")
    print(f"  行動次元: {action_space.shape if hasattr(action_space, 'shape') else 'N/A'}")
except Exception as e:
    print(f"⚠️  観測・行動空間の取得に失敗: {e}")

# ポリシー（行動関数）の設定
print("\n" + "=" * 70)
print("🎯 ポリシー（行動関数）の設定")
print("=" * 70)

fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
fcnet_activation = model_config.get("fcnet_activation", "tanh")
vf_share_layers = model_config.get("vf_share_layers", True)

print(f"\n  隠れ層のサイズ: {fcnet_hiddens}")
print(f"  隠れ層の数: {len(fcnet_hiddens)}")
print(f"  活性化関数: {fcnet_activation}")

# 各層の詳細
print(f"\n  各層の詳細:")
for i, hidden_size in enumerate(fcnet_hiddens):
    print(f"    層 {i+1}: {hidden_size} ユニット (活性化: {fcnet_activation})")

# 価値関数の設定
print("\n" + "=" * 70)
print("💰 価値関数の設定")
print("=" * 70)

print(f"\n  層の共有: {vf_share_layers}")

if vf_share_layers:
    print(f"  → ポリシーと価値関数は同じ隠れ層を共有")
    print(f"  → 隠れ層: {fcnet_hiddens}")
    print(f"  → 活性化関数: {fcnet_activation}")
else:
    vf_hiddens = model_config.get("vf_hiddens", [256, 256])
    vf_activation = model_config.get("vf_activation", fcnet_activation)
    print(f"  → 価値関数専用の隠れ層を使用")
    print(f"  → 隠れ層のサイズ: {vf_hiddens}")
    print(f"  → 隠れ層の数: {len(vf_hiddens)}")
    print(f"  → 活性化関数: {vf_activation}")
    print(f"\n  各層の詳細:")
    for i, hidden_size in enumerate(vf_hiddens):
        print(f"    層 {i+1}: {hidden_size} ユニット (活性化: {vf_activation})")

# その他のモデル設定
print("\n" + "=" * 70)
print("⚙️  その他のモデル設定")
print("=" * 70)

other_settings = {
    "fcnet_activation": model_config.get("fcnet_activation"),
    "no_final_linear": model_config.get("no_final_linear", False),
    "free_log_std": model_config.get("free_log_std", False),
    "vf_share_layers": model_config.get("vf_share_layers", True),
    "use_lstm": model_config.get("use_lstm", False),
    "max_seq_len": model_config.get("max_seq_len", None),
    "lstm_cell_size": model_config.get("lstm_cell_size", None),
    "lstm_use_prev_action": model_config.get("lstm_use_prev_action", False),
    "lstm_use_prev_reward": model_config.get("lstm_use_prev_reward", False),
}

for key, value in other_settings.items():
    if value is not None:
        print(f"  {key}: {value}")

# アルゴリズム固有の設定
print("\n" + "=" * 70)
print("📚 アルゴリズム設定")
print("=" * 70)

algo_name = algo.__class__.__name__
print(f"\n  アルゴリズム: {algo_name}")

if hasattr(algo.config, "lr"):
    print(f"  学習率: {algo.config.lr}")
if hasattr(algo.config, "gamma"):
    print(f"  割引率 (gamma): {algo.config.gamma}")
if hasattr(algo.config, "lambda_"):
    print(f"  GAE lambda: {algo.config.lambda_}")

# 完全なモデル設定（デバッグ用）
print("\n" + "=" * 70)
print("📋 完全なモデル設定（詳細）")
print("=" * 70)
print("\n")
pprint(dict(model_config), width=80, indent=2)

print("\n" + "=" * 70)
print("✅ 設定確認完了")
print("=" * 70)

ray.shutdown()
