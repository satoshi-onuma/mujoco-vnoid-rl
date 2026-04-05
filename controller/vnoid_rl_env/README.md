# Choreonoid End-to-End RL 環境

Choreonoid シミュレータ上で 29 関節の二足歩行ロボットを  
End-to-End 強化学習 (RL) するための環境です。

## アーキテクチャ

```
Python プロセス                    C++ プロセス (Choreonoid)
──────────────────────             ─────────────────────────────
RLlib / Gymnasium                  VnoidRLController
ChoreonoidShmClient                ShmInterface (create=false)
       │                                   │
       └──── POSIX 共有メモリ ─────────────┘
             /dev/shm/vnoid_rl_shm_<env_id>
```

### 共有メモリレイアウト (824 bytes)

| offset | ブロック | サイズ | 方向 | 内容 |
|-------:|---------|------:|------|------|
| 0      | ControlBlock | 24 B | 双方向 | 制御フラグ (step/reset/ready/done) + 報酬 |
| 24     | ActionBlock  | 232 B | Python → C++ | 全関節目標角度 (29 × double) |
| 256    | StateBlock   | 568 B | C++ → Python | 観測ベクトル (71 × double) |

### 観測空間 (71 次元, float64)

| インデックス | 内容 | 単位 |
|------------|------|------|
| 0–2   | ルートリンク位置 (x, y, z)                | m      |
| 3–6   | ルートリンク姿勢クォータニオン (w, x, y, z) | —      |
| 7–9   | ルートリンク線速度 (x, y, z)              | m/s    |
| 10–12 | ルートリンク角速度 (x, y, z)              | rad/s  |
| 13–41 | 全関節角度 (29 joints)                   | rad    |
| 42–70 | 全関節角速度 (29 joints)                 | rad/s  |

### 行動空間 (29 次元, float32, [-1, 1])

各関節の正規化目標角度。C++ 側で関節可動域にスケールされ PD 制御に渡す。

---

## セットアップ

### 1. micromamba 環境の作成

```bash
# 環境作成 (初回のみ)
micromamba create -n vnoid_rl -c conda-forge \
    python=3.11 numpy gymnasium pytorch posix_ipc ray-default

# 環境をアクティベート
micromamba activate vnoid_rl
```

> **パッケージ名の対応**: conda-forge では `torch` → `pytorch`、`ray[rllib]` → `ray-default` という名前になる。

既存環境に追加インストールする場合:
```bash
micromamba install -n <env_name> -c conda-forge \
    numpy gymnasium pytorch posix_ipc ray-default
```

### 2. Choreonoid コントローラのビルド

```bash
cd ~/choreonoid/build
cmake .. -DVNOID_BUILD_CNOID=ON
make vnoid_rl_controller -j$(nproc)
```

---

## 使い方

### Step 1: 疎通テスト

共有メモリの接続確認を手動で行う場合の手順。

**ターミナル 1 (Python 側):**
```bash
cd ~/choreonoid/ext/vnoid/controller/vnoid_rl_env
python shm_interface.py
# 「共有メモリ作成完了」と表示されたら Choreonoid を起動する
```

**ターミナル 2 (Choreonoid 側):**
```bash
ENV_ID=0 ~/choreonoid/build/bin/choreonoid \
    ~/choreonoid/ext/vnoid/project/vnoid_rl_project.cnoid \
    --start-simulation
```

### Step 2: Gymnasium 環境のテスト

ターミナル 1 で python choreonoid_env.py を実行
「共有メモリ作成完了」が表示されたら
ターミナル 2 で Choreonoid を起動


```bash
python choreonoid_env.py
# ランダム行動で 10 ステップ実行し結果を表示
```

### Step 3: RL 学習

`train_choreonoid.py` は Choreonoid プロセスを **自動で起動・終了** する。
Choreonoid を別途起動する必要はない。

```bash
python train_choreonoid.py
```

実行時の流れ:
1. 共有メモリを作成
2. `subprocess.Popen` で `ENV_ID=0,1,...` を付けて Choreonoid を自動起動
3. 学習ループ (100 イテレーション)
4. 終了時に Choreonoid プロセスを自動終了

---

## 同期プロトコル

1 ステップの通信シーケンス:

```
Python                      C++ (Choreonoid control ループ)
──────────────────────      ───────────────────────────────────
1. ready = 0         →
2. action を書き込み →
3. step_request = 1  →      4. step_request を検知
                             5. PD 制御実行
                             6. obs / reward / done を書き込み
                             7. step_request = 0
               ←             8. ready = 1  ← (メモリバリア後)
9. ready == 1 を確認
10. obs / reward / done を読み取り
```

> **重要**: Python は手順 1 で `ready = 0` にクリアしてから
> リクエストを送らなければならない。これを省略すると前回の
> `ready=1` が残り、C++ の処理完了前に Python が即リターンする
> (旧実装のバグ)。

---

## ファイル構成

```
controller/
├── vnoid_rl_controller/       # C++ SimpleController
│   ├── main.cpp               # コントローラ本体 (制御ループ・報酬・終了判定)
│   ├── shm_interface.h        # 共有メモリ構造体・ShmInterface クラス
│   └── CMakeLists.txt
│
└── vnoid_rl_env/              # Python RL 環境
    ├── shm_interface.py       # 共有メモリクライアント (ChoreonoidShmClient)
    ├── choreonoid_env.py      # Gymnasium 環境 (ChoreonoidEnv)
    ├── train_choreonoid.py    # 学習スクリプト
    ├── test_connection.py     # 疎通テスト
    └── README.md              # このファイル
```

---

## 並列実行

`train_choreonoid.py` の `NUM_WORKERS` を増やすだけで並列化できる。
Choreonoid の起動・終了はスクリプトが自動管理する。

```python
# train_choreonoid.py
NUM_WORKERS = 8  # この数だけ Choreonoid プロセスが自動起動される
```

### env_id の対応

Choreonoid プロセスと Ray ワーカーは共有メモリ名 `/vnoid_rl_shm_<env_id>` で 1対1 に対応する。

```
ENV_ID=0 Choreonoid ←→ /vnoid_rl_shm_0 ←→ Ray worker_index=0
ENV_ID=1 Choreonoid ←→ /vnoid_rl_shm_1 ←→ Ray worker_index=1
...
```

`make_choreonoid_env` は `env_id = worker_index * num_envs_per_worker + vector_index` で
env_id を計算する。`num_envs_per_env_runner=1`（固定）のとき `env_id = worker_index` となり、
Choreonoid の `ENV_ID` と一致する。

> **注意**: `num_envs_per_env_runner` を 1 以外に変更すると env_id がずれるため禁止。

---

## トラブルシューティング

### 共有メモリが残留している

```bash
rm -f /dev/shm/vnoid_rl_shm_*
```

### Choreonoid が応答しない (タイムアウト)

- `ENV_ID` 環境変数が Python 側と一致しているか確認する
- Choreonoid のコントローラログにエラーが出ていないか確認する

### ビルドエラー

```bash
cd ~/choreonoid/build
rm CMakeCache.txt
cmake .. -DVNOID_BUILD_CNOID=ON
make vnoid_rl_controller -j$(nproc)
```

---

## 今後の予定

1. ✅ 疎通確認 (1 環境)
2. ⬜ 学習開始 (1 環境)
3. ⬜ 並列化 (8 環境)
4. ⬜ AGX Dynamics 移行
