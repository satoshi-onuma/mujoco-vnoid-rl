# vnoid_rl_env

MuJoCo + Vnoid 歩行制御を Python から使うための pybind11 モジュールです。  
Gymnasium 互換の RL 環境 `VnoidEnv` を提供します。

## ファイル構成

```
controller/vnoid_rl_env/
├── README.md                 # このファイル
├── CMakeLists.txt            # ビルド設定
├── bindings.cpp              # Python モジュール定義（pybind11）
├── vnoid_env.h               # VnoidEnv クラス宣言
├── vnoid_env_lifecycle.cpp   # 初期化・終了処理
├── vnoid_env_episode.cpp     # エピソード制御（reset / step）
├── vnoid_env_reward.cpp      # 報酬・観測
├── vnoid_env_terrain.cpp     # 地盤パラメータ切替
├── vnoid_env_logging.cpp     # CSV ログ出力
├── vnoid_env_render.cpp      # レンダリング・GUI 操作
├── myrobot.h                 # Vnoid ロボット制御クラス宣言
└── myrobot.cpp               # Vnoid ロボット制御クラス実装
```

## 各ファイルの役割

### `bindings.cpp`

Python から見える API を定義するエントリポイントです。  
`VnoidEnv` クラスを `vnoid_rl_env` モジュールとして公開します。

公開メソッド:

| メソッド | 説明 |
|---|---|
| `reset()` | 環境を初期状態に戻し、観測を返す |
| `step(action)` | 1 歩分の制御を実行し、(obs, reward, terminated, info, frames) を返す |
| `get_observation()` | 現在の観測ベクトルを返す |
| `set_walk_command(stride, sway, turn)` | 歩容コマンド（stride / sway / turn）を設定 |
| `set_reward_weights(w_track, w_act, w_healthy, tracking_sigma)` | 報酬重みを設定 |
| `set_terrain_config(cfg)` | 歩行途中で切り替わる先の地盤設定を辞書で受け取る（保持のみ、適用は `step()` 内） |
| `should_close()` | レンダリングウィンドウが閉じられたか |
| `get_control_log()` | 後方互換用（空 dict を返す） |
| `clear_control_log()` | 後方互換用（何もしない） |

### `vnoid_env.h`

`VnoidEnv` クラスの宣言と、分割された各 `.cpp` ファイル間で共有する型定義を置きます。

主な内容:

- `FootState` 構造体（足の位置・速度）
- `VnoidEnv` のメンバ変数・メソッド宣言
- `VNOID_REWARD_LOG_DEBUG` マクロ（報酬検証用ログの ON/OFF）

### `vnoid_env_lifecycle.cpp`

環境の生成から破棄までのライフサイクルを担当します。

| 関数 | 説明 |
|---|---|
| コンストラクタ | MuJoCo モデル読み込み、ロボット初期化、レンダリング有効時は CSV / OpenGL 初期化 |
| デストラクタ | CSV クローズ、cleanup 呼び出し |
| `initFootBodyIds()` | MuJoCo 上の足 body ID（`R_FOOT_R`, `L_FOOT_R`）を取得 |
| `initializeRobot()` | ロボットコントローラの再生成 |
| `cleanup()` | MuJoCo / GLFW / レンダリングリソースの解放 |

### `vnoid_env_episode.cpp`

RL エピソードの進行を担当します。

| 関数 | 説明 |
|---|---|
| `reset()` | `mjData` とロボットを再初期化、地盤を硬地盤に戻し、観測を返す |
| `set_walk_command()` | 外部から歩容指令を注入し、footstep 計画を更新 |
| `step()` | RL アクションを受け取り、1 歩完了まで MuJoCo シミュレーションを回す。転倒判定・地盤切替・ログ記録・レンダリングもここから呼ばれる |

### `vnoid_env_reward.cpp`

観測と報酬の計算を担当します。

| 関数 | 説明 |
|---|---|
| `get_observation()` | 角速度・姿勢・加速度・接触・足沈み込み・歩容コマンドなどを観測ベクトルにまとめる |
| `compute_reward()` | tracking / healthy / action penalty から報酬を計算 |
| `reward_tracking()` | 1 歩あたりの stride / sway 追従誤差に基づく報酬 |
| `reward_action_penalty()` | RL アクション（foot offset）の大きさに対するペナルティ |
| `reward_healthy()` | 生存報酬（常に 1.0） |
| `log_reward_step()` | `VNOID_REWARD_LOG_DEBUG` 有効時、報酬内訳を `reward_log.csv` に出力 |

### `vnoid_env_terrain.cpp`

MuJoCo の床（`floor` geom）の接触パラメータを変更し、ドメインランダマイゼーションを行います。

| 関数 | 説明 |
|---|---|
| `apply_terrain()` | friction / solref / solimp を直接設定 |
| `apply_hard_terrain()` | 硬地盤プリセット |
| `apply_soft_terrain()` | 軟地盤プリセット |
| `apply_debug_terrain()` | デバッグ用地盤プリセット |
| `apply_random_terrain()` | 乱数で地盤パラメータを生成して適用 |
| `set_terrain_config(cfg)` | 切り替え先の設定を保持するだけ（Python から呼ぶ） |
| `apply_switch_terrain()` | 保持した設定に従って切り替えを実行（`step()` から呼ばれる） |

`reset()` 時に硬地盤から開始し、`terrain_switch_at`（reset ごとに抽選）に達した時点で
`step()` 内から `apply_switch_terrain()` が呼ばれ、歩行途中で地盤が変わります。
切り替え先だけ Python から `set_terrain_config()` で指定でき、切り替えタイミングは C++ が持ちます。

### `vnoid_env_logging.cpp`

レンダリング有効時の制御データ CSV 出力を担当します。

| 関数 | 説明 |
|---|---|
| `get_control_log()` / `clear_control_log()` | 後方互換用スタブ |
| `get_foot_state()` | MuJoCo から足 body の位置・速度を取得 |
| `calc_com_velocity()` | CoM 速度を数値微分で計算 |
| `calc_dcm_actual()` | 実測 DCM を計算 |
| `calc_angular_momentum_around_com()` | 重心周り角運動量を MuJoCo から取得 |
| `log_control_data()` | 上記を含む制御量を `control_log.csv` に 1 行書き込む |

### `vnoid_env_render.cpp`

OpenGL による可視化と録画フレーム生成、および GUI 操作を担当します。

| 関数 | 説明 |
|---|---|
| `initializeGLFW()` | ウィンドウ作成、キー/マウスコールバック登録 |
| `initializeRenderer()` | MuJoCo シーン・コンテキスト初期化 |
| `updateDisplay()` | 画面への描画とイベント処理 |
| `render()` | 録画用 1280x720 フレームを numpy 配列として返す |
| `keyboard()` | Backspace で reset |
| `mouse_button()` / `mouse_move()` / `scroll()` | カメラ操作 |

### `myrobot.h` / `myrobot.cpp`

Vnoid 歩行制御の本体です。`RobotMujoco` を継承し、footstep 計画・安定化制御・IK/FK を MuJoCo 上で実行します。

| 主な API | 説明 |
|---|---|
| `Init(m, d)` | MuJoCo モデル/データで初期化 |
| `Control(rl_params)` | 1 制御サイクル分の vnoid 制御 + RL 介入 |
| `SetWalkCommand()` | stride / sway / turn を設定 |
| `UpdateFootstepPlan()` | 現在の walk_cmd で footstep を再計画 |

## データフロー（概要）

```
Python (train_humanoid.py 等)
    │
    ▼
bindings.cpp          … pybind11 経由で VnoidEnv を公開
    │
    ▼
vnoid_env_episode.cpp … reset / step
    │
    ├─► myrobot.cpp         … vnoid 歩行制御
    ├─► vnoid_env_reward.cpp … 観測・報酬
    ├─► vnoid_env_terrain.cpp … 地盤切替
    ├─► vnoid_env_logging.cpp … CSV ログ
    └─► vnoid_env_render.cpp  … 描画（レンダリング有効時）
```

## ビルド

```bash
cmake ..   -DVNOID_BUILD_MUJOCO=ON   -DVNOID_BUILD_CNOID=OFF   -DCMAKE_INSTALL_PREFIX=~/vnoid-mujoco-install   -DCMAKE_PREFIX_PATH=~/mujoco-install

make -j8
```


## Python からの利用例

```python
import vnoid_rl_env

env = vnoid_rl_env.VnoidEnv("model/sample_robot/sample_robot_mujoco.xml")
obs = env.reset()
env.set_walk_command(0.1, 0.0, 0.0)
obs, reward, terminated, info, frames = env.step([0.0, 0.0])
```

関連スクリプト:

- `python_scripts/my_humanoid_env.py` … Gymnasium ラッパ
- `python_scripts/train_humanoid.py` … 学習スクリプト
- `python_scripts/plot_control_data.py` … `control_log.csv` の可視化
