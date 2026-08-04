# RL Experiment Manager (tkinter)

期末レポートで提案した実験管理アプリのデスクトップ実装です。
ブラウザの Web UI ではなく、ローカルで動く tkinter GUI です。

## できること

- 報酬重み・地盤・PPOハイパーパラメータをフォームから指定して学習を起動
- 実験ごとのひとことメモ（任意）を保存・一覧表示
- 単発実験の起動・停止
- 学習中の報酬曲線をリアルタイム表示
- 履歴タブからの手動評価（複数地盤で録画・歩行距離記録）
- 過去実験の一覧・動画再生

## ファイル構成

```
rl_experiment_app/
├── README.md              # このファイル
├── __init__.py            # パッケージ宣言（中身なし）
├── main.py                # アプリ起動・タブ接続・学習プロセス監視
├── database.py            # SQLite による実験・評価データの CRUD
├── training_launcher.py   # train_humanoid.py の subprocess 起動・監視
├── eval_launcher.py       # record_humanoid.py を呼び出して評価を実行
└── ui/
    ├── __init__.py
    ├── run_tab.py         # 実行タブ（パラメータ入力 / Run・Stop）
    ├── progress_tab.py    # 進捗タブ（学習曲線のリアルタイム表示）
    └── history_tab.py     # 履歴タブ（一覧 / 評価実行 / 動画再生）
```

アプリ外で実際の学習・録画を担うスクリプト（このアプリから呼び出される）:

| ファイル | 役割 |
|---|---|
| `python_scripts/train_humanoid.py` | Ray RLlib PPO 学習。CLI引数で重み・地盤・出力先を受け取り、`~/vnoid-experiments/runs/<run_id>/` に CSV・チェックポイント・`result.json` を書く |
| `python_scripts/record_humanoid.py` | 学習済み方策の録画・評価。チェックポイントを読み、動画とログを `run_dir` に書き、歩行距離などを `EVAL_RESULT_JSON:` で標準出力する |
| `python_scripts/my_humanoid_env.py` | Gymnasium ラッパ。`reward_weights` / `terrain_config` を C++ の `VnoidEnv` に渡す |

## 各ファイルの役割

### `main.py`

エントリポイント。`tk.Tk` を立て、`ttk.Notebook` で3タブを並べる。

| 担当 | 内容 |
|---|---|
| `ExperimentApp` | DB・Launcher・EvalLauncher を生成し、各タブに配線する |
| `start_single()` | 実行タブの Run から呼ばれ、学習を起動して進捗監視を開始する |
| `stop_all()` | 学習プロセスを停止する |
| `_poll()` | 1秒ごとに CSV を再描画し、プロセス終了を検知する |

起動: `python -m rl_experiment_app.main`

### `database.py`

実験メタデータと評価結果を SQLite（`~/vnoid-experiments/runs/experiments.db`）に保存する。

| テーブル | 内容 |
|---|---|
| `experiments` | run_id、設定（報酬重み・地盤・HP）、状態、チェックポイント/CSVパス、最終報酬、メモなど |
| `evaluations` | 実験ごとの評価結果（地盤、歩行距離、動画・ログパス） |

主な API: `insert_experiment` / `update_experiment_status` / `insert_evaluation` / `list_experiments` / `get_experiment`

### `training_launcher.py`

GUI から `train_humanoid.py` を `subprocess.Popen` で起動し、終了を監視する。

| 担当 | 内容 |
|---|---|
| `make_run_id()` | 日時＋短い UUID で run_id を生成 |
| `TrainingLauncher.start()` | `runs/<run_id>/` を作り、DB に `running` で登録して学習を起動。stdout は `train.log` へ |
| `poll()` | プロセス終了を検知。成功時は `result.json` を読んで DB を `completed` に更新 |
| `stop()` | `terminate` / 必要なら `kill`。DB は `early_stopped` |

### `eval_launcher.py`

履歴タブの「評価を実行」から呼ばれ、`record_humanoid.py` を地盤ごとに実行する。

| 担当 | 内容 |
|---|---|
| `evaluate_run()` | 指定 run のチェックポイントで、既定の切り替え先地盤（soft / random）ごとに録画 |
| `_run_record()` | subprocess で録画し、`EVAL_RESULT_JSON:` 行を読んで `evaluations` テーブルに保存 |

### `ui/run_tab.py`

実行タブ。報酬重み・切替先地盤・PPOハイパーパラメータの入力フォームと Run / Stop ボタン。

### `ui/progress_tab.py`

進捗タブ。`training_stats.csv` を定期読み込みし、matplotlib（TkAgg）で学習曲線を表示する。

### `ui/history_tab.py`

履歴タブ。過去実験の Treeview 一覧、「評価を実行」「動画を再生」ボタン。
選択行のメモを下部の入力欄で編集・追記でき、「メモ保存」で DB に書き戻す。
評価は別スレッドで回し、GUI が固まらないようにする。動画は `xdg-open` で OS 既定プレイヤーを起動する。

## データフロー（概要）

```
[実行タブ] Run
    │
    ▼
training_launcher.py  ──subprocess──► train_humanoid.py
    │                                      │
    │                                      ├─► runs/<run_id>/training_stats.csv
    │                                      ├─► runs/<run_id>/checkpoint/
    │                                      └─► runs/<run_id>/result.json
    │
    ├─► database.py (experiments: running → completed)
    └─► [進捗タブ] CSV をポーリングしてグラフ更新

[履歴タブ] 評価を実行
    │
    ▼
eval_launcher.py  ──subprocess──► record_humanoid.py
    │                                      │
    │                                      └─► 動画 / recording_log / EVAL_RESULT_JSON
    └─► database.py (evaluations に記録)
```

## データの保存場所

生成物はすべてリポジトリ外に置きます。

```
~/vnoid-experiments/
  archive/          # 過去実験の退避先
  runs/
    experiments.db  # SQLite
    <run_id>/
      checkpoint/
      training_stats.csv
      result.json
      train.log
      soft_demo.mp4
      ...
```

## 起動方法

環境は micromamba の `robot_env` を使います。

```bash
micromamba activate robot_env

# C++ 環境モジュールのビルド（報酬重み setter 等を追加した後は再ビルドが必要）
cd build
cmake ..   -DVNOID_BUILD_MUJOCO=ON   -DVNOID_BUILD_CNOID=OFF   -DCMAKE_INSTALL_PREFIX=~/vnoid-mujoco-install   -DCMAKE_PREFIX_PATH=~/mujoco-install
make -j8
cd ..

# GUI 起動
python -m rl_experiment_app.main
```

## CLI からも従来どおり学習可能

```bash
cd python_scripts
python train_humanoid.py --run-id my_exp --w-track 1.2 --terrain soft --num-iterations 50
python record_humanoid.py --checkpoint-dir ~/vnoid-experiments/runs/my_exp/checkpoint \
  --run-dir ~/vnoid-experiments/runs/my_exp --terrain soft
```

引数なしの `python train_humanoid.py` は `~/vnoid-experiments/runs/adhoc_<timestamp>/` に出力します。

## 評価

学習終了時に評価は自動実行されません。履歴タブで完了済みの実験を1件選び、
「評価を実行」を押すと、切り替え先地盤ごとの録画とログ取得を開始します。

スキーマを変更した場合は、古いDBを削除してから再起動してください。

```bash
rm ~/vnoid-experiments/runs/experiments.db
```

## 地盤（terrain）の扱い

エピソードは**必ず硬地盤から始まり**、C++側が決めたランダムなタイミング（`reset()` 内で
`terrain_switch_at` を抽選）で地盤が切り替わります。この「歩行途中で地面が変わる」挙動は
研究の本質なので、Python 側からは切り替えを行いません。

`--terrain` で指定できるのは**切り替え先**の地盤です。Python は設定値の辞書を
`set_terrain_config()` で C++ に渡すだけで、適用は C++ の切り替えタイミングで行われます。

```python
env = HumanoidVnoidEnv(terrain_config={"mode": "soft"})
# 明示パラメータを渡すことも可能
env = HumanoidVnoidEnv(terrain_config={
    "friction": 1.0, "solref0": 0.1, "solref1": 2.0,
    "solimp0": 0.7, "solimp1": 0.85, "solimp2": 0.003,
})
```
