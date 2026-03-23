# Choreonoid End-to-End RL環境

Choreonoid + AIST Simulatorで完全End-to-End強化学習を行うための環境です。

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install posix_ipc numpy gymnasium ray[rllib] torch
```

### 2. Choreonoidのビルド

```bash
cd ~/choreonoid/build
cmake .. -DVNOID_BUILD_CNOID=ON
make vnoid_rl_controller -j8
```

## 使い方

### Step 1: 疎通テスト

まず、共有メモリが正常に動作するか確認します。

**ターミナル1（Python側）:**
```bash
cd ~/choreonoid/ext/vnoid/controller/vnoid_rl_env
python test_connection.py
```

**ターミナル2（Choreonoid側）:**
```bash
ENV_ID=0 ~/choreonoid/build/bin/choreonoid ~/choreonoid/ext/vnoid/project/vnoid_rl_project.cnoid --start-simulation
```

疎通に成功すると、Python側で「疎通テスト成功！」と表示されます。

### Step 2: 環境クラスのテスト

```bash
cd ~/choreonoid/ext/vnoid/controller/vnoid_rl_env
python choreonoid_env.py
```

ランダム行動で10ステップ実行されます。

### Step 3: RL学習

```bash
cd ~/choreonoid/ext/vnoid/controller/vnoid_rl_env
python train_choreonoid.py
```

自動的にChoreonoidプロセスが起動し、学習が開始されます。

## アーキテクチャ

```
Python (RLlib PPO) <--共有メモリ--> Choreonoid SimpleController
                                         ↓
                                   AIST Simulator
```

- **観測空間**: 73次元（センサー生値全部）
- **行動空間**: 29次元（全関節目標角度）
- **制御周期**: 1ms (1000Hz)
- **学習アルゴリズム**: PPO

## ファイル構成

```
controller/
├── vnoid_rl_controller/       # C++ SimpleController
│   ├── main.cpp               # コントローラ本体
│   ├── shm_interface.h        # 共有メモリインターフェース
│   └── CMakeLists.txt
│
├── vnoid_rl_env/              # Python RL環境
│   ├── shm_interface.py       # 共有メモリクライアント
│   ├── choreonoid_env.py      # Gymnasium環境
│   ├── train_choreonoid.py    # 学習スクリプト
│   └── test_connection.py     # 疎通テスト
│
project/
└── vnoid_rl_project.cnoid     # Choreonoidプロジェクトファイル
```

## トラブルシューティング

### 共有メモリエラー

共有メモリが残っている場合：
```bash
rm /dev/shm/vnoid_rl_shm_*
```

### Choreonoidが起動しない

環境変数が正しく設定されているか確認：
```bash
echo $ENV_ID
```

### ビルドエラー

CMakeを再実行：
```bash
cd ~/choreonoid/build
rm CMakeCache.txt
cmake .. -DVNOID_BUILD_CNOID=ON
make vnoid_rl_controller -j8
```

## 次のステップ

1. ✅ 疎通確認（1環境）
2. ⬜ 学習開始（1環境）
3. ⬜ 並列化（8環境）
4. ⬜ AGX移行
