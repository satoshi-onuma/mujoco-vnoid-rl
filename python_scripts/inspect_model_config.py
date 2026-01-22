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
    algo = Algorithm.from_checkpoint(checkpoint_dir
    )
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

# ---------------------------------------------------------
# 追加: 学習の進捗・時間情報の確認
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("⏱️ 学習進捗・時間の確認 (Internal State)")
print("=" * 70)

# アルゴリズムの全状態を取得
state = algo.get_state()

# 1. 内部属性からの取得 (Rayの多くのバージョンで有効な非公式属性)
# "_time_total_s" は学習の累積時間を保持していることが多いです
internal_time = getattr(algo, "_time_total_s", None)
if internal_time is not None:
    print(f"\n【内部属性 (_time_total_s)】")
    print(f" 累積学習時間: {internal_time:.2f} 秒 ({internal_time/3600:.2f} 時間)")
else:
    print("\n【内部属性】 _time_total_s は見つかりませんでした。")

# 2. Counters (ステップ数などの累積カウンター) の確認
# Ray 2.x系ではここに 'num_env_steps_sampled' などが含まれます
if "counters" in state:
    print("\n【Counters (累積ステップ数など)】")
    counters = state["counters"]
    
    # 主要なカウンターをピックアップして表示
    target_keys = [
        "num_env_steps_sampled", 
        "num_env_steps_sampled_lifetime",
        "num_agent_steps_sampled",
        "num_agent_steps_trained"
    ]
    
    found_any = False
    for key in target_keys:
        if key in counters:
            print(f" {key}: {counters[key]}")
            found_any = True
            
    # 特定のキーが見つからない場合は全表示（デバッグ用）
    if not found_any:
        print(" 主要なカウンターキーが見つかりません。全countersを表示します:")
        pprint(counters, width=80, indent=2)
else:
    print("\n【Counters】 状態辞書に 'counters' が含まれていません。")

# 3. Timers (処理時間の計測記録) の確認
# 学習ループ内の各処理にかかった時間などが記録されている場合があります
if "timers" in state:
    print("\n【Timers (処理時間統計)】")
    # 情報量が多い場合があるので、学習時間に関連しそうなものがあれば表示
    timers = state["timers"]
    if timers:
        pprint(timers, width=80, indent=2)
    else:
        print(" (空のデータ)")


# ---------------------------------------------------------
# 4. デバッグ: Stateの中身を総点検する（これが一番確実です）
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🕵️ State辞書のキー総点検")
print("=" * 70)

state = algo.get_state()

# トップレベルのキーをすべて表示
print(f"State keys: {list(state.keys())}")

# 【最有力候補】 global_vars (Ray 2.x系の古いチェックポイントはここに情報があることが多い)
if "global_vars" in state:
    print("\n【global_vars】 (ここに timesteps_total があるはずです)")
    pprint(state["global_vars"], width=80, indent=2)

# 【次点】 timesteps
if "timesteps" in state:
    print(f"\n【timesteps】: {state['timesteps']}")

# 【次点】 info (統計情報が入ることがある)
if "info" in state:
    print("\n【info】")
    pprint(state["info"], width=80, indent=2)

print("\n" + "=" * 70)

# ---------------------------------------------------------
# 5. New API Stack専用: MetricsLoggerとIterationの解析
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🆕 New API Stack データ解析")
print("=" * 70)

# (1) 学習回数 (Training Iterations)
# 新しいスタックでは、ここには単純な整数（回数）が入っていることが多いです
if "training_iteration" in state:
    print(f"\n【学習回数 (training_iteration)】: {state['training_iteration']}")

# (2) MetricsLogger の中身を確認
# ここに学習時間 (time_total_s) や 生涯ステップ数 (num_env_steps_sampled_lifetime) が入っています
if "metrics_logger" in state:
    print("\n【MetricsLogger (主要統計データ)】")
    ml_state = state["metrics_logger"]
    
    # MetricsLoggerの状態は通常、以下のような辞書構造になっています
    # {
    #   "stats": {
    #       "time_total_s": { ... value ... },
    #       "num_env_steps_sampled_lifetime": { ... value ... }
    #   }
    # }
    
    # 構造が深いため、'stats' キー以下を探索します
    stats = ml_state.get("stats", {})
    
    # 時間・ステップ数に関連しそうなキーを探して表示
    target_metrics = [
        "time_total_s",
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled", 
        "num_agent_steps_sampled_lifetime"
    ]
    
    found_metric = False
    for key, val in stats.items():
        # キー名に target_metrics のいずれかが含まれていれば表示
        if any(t in key for t in target_metrics):
            print(f" 🔹 {key}: {val}")
            found_metric = True
            
    if not found_metric:
        print(" (主要なメトリクスが見つかりませんでした。全データを表示します)")
        pprint(ml_state, width=80, indent=2)

print("\n" + "=" * 70)
ray.shutdown()
