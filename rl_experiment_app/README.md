# RL Experiment Manager (tkinter)

ローカルで動く tkinter GUI の実験管理アプリです。

## できること

- 報酬重み・地盤・PPOハイパーパラメータをフォームから指定して学習をキュー投入
- 学習キューと評価キューを**別々に**管理（学習中は eval 1 本まで、**学習 idle 時は eval 最大8本**同時、実行タブで1〜8に変更可）
- 学習中の報酬曲線を進捗タブでリアルタイム表示
- 履歴から softness 格子評価（`eval_softness_grid` / `eval_softness_paper_log`）を評価キューへ投入
- Plot タブで成功率・seed平均エピソード長・paper control・sink vs Δx・Δx vs overall 成功を比較描画
- 実験メモの保存・一覧・削除

動画付き評価（`record_humanoid`）は UI から外しています。必要なときは CLI で手動実行してください。

## ファイル構成

```
rl_experiment_app/
├── README.md
├── __init__.py
├── main.py                     # 二重キュー・共通 poll
├── database.py
├── training_launcher.py        # train_humanoid.py
├── softness_eval_launcher.py   # eval_softness_grid / paper_log
├── eval_pool.py                # 複数 eval subprocess
├── eval_launcher.py            # 旧動画評価（アプリ未使用・CLI参考用）
└── ui/
    ├── run_tab.py              # 学習フォーム + train/eval 待ちキュー
    ├── progress_tab.py         # ライブ学習曲線
    ├── history_tab.py          # 一覧 / 評価キュー投入 / 削除
    └── plot_tab.py             # Softness成功 / 学習曲線 / Paper control / Paper比較
```

呼び出されるスクリプト:

| ファイル | 役割 |
|---|---|
| `python_scripts/train_humanoid.py` | PPO 学習 → `~/vnoid-experiments/runs/<run_id>/` |
| `python_scripts/eval_softness_grid.py` | softness×seed 成功率 → `~/vnoid-experiments/evals/` |
| `python_scripts/eval_softness_paper_log.py` | 成功率 + control.csv → `~/vnoid-experiments/paper_logs/` |
| `python_scripts/plot_softness_success.py` | 成功率比較 Plot |
| `python_scripts/plot_training_curves_seeds.py` | seed平均エピソード長 |
| `python_scripts/plot_paper_control.py` | paper control 4段図 / 複数 trial 7パネル比較 |
| `python_scripts/plot_deltax_success.py` | チェックポイント Δx平均 vs overall 成功率 |
| `python_scripts/record_humanoid.py` | 手動録画用（UI 非接続） |

## タブ

| タブ | 内容 |
|---|---|
| 実行 | 学習パラメータ、学習待ち／評価待ちキュー、Train Stop / Eval Stop |
| 進捗 | 実行中 train のライブ曲線 |
| 履歴 | 実験一覧、softness 評価をキューへ、メモ、削除 |
| Plot | Softness成功 / 学習曲線 / Paper control / Paper比較 |

## キューの動き

- **学習キュー**と**評価キュー**は独立（メモリ上のみ、アプリ終了で消える）
- 各キューは直列に**投入**するが、評価は idle 時に複数本を同時実行できる
- 学習実行中は評価同時数 **1**（train + eval 1 本は従来どおり共存）
- 学習 idle 時は評価同時数 **K**（既定 **8**）。同じ `output_dir` のジョブは同時に1本だけ（CSV 競合防止）
- Stop は種別ごと（今の1本だけ止め、待ちは残す）
- 失敗しても次の予約へ進む

起動: `python -m rl_experiment_app.main`

## 評価（履歴 → 評価キュー）

1. 完了／early_stopped で checkpoint がある run を選択（複数可）
2. seeds / softness min·max·step / script（`grid` または `paper_log`）を指定
3. 「評価をキューへ」

初期表示例: seeds `1001-1010`、softness `0.0 / 1.2 / 0.1`、script `grid`。フォームでいつでも変更可。

同条件の再実行は既存出力ディレクトリを上書きします（確認ダイアログなし）。ログは `~/vnoid-experiments/evals/_logs/`。

## Plot タブ

| モード | データ | 操作 |
|---|---|---|
| Softness成功 | `evals/*/summary.csv` と `paper_logs/*/summary.csv` | ラベル付きグループ化 → mean±std / 学習シード別細線（切替）。一覧は `[evals]` / `[paper]` で区別 |
| 学習曲線 | DB 実験の `training_stats.csv` | 同条件 seed をグループ化 → episode_len mean±std |
| Paper control | `paper_logs/<run>/seed*_softness_*` | 1 trial を4段表示 |
| Paper比較 | 同上（複数 trial） | ラベル付き比較リスト → 7信号パネルに色分け重ね描き。PNG保存時は `{ラベル}_control.csv` も同ディレクトリへ |
| Δx vs 成功 | DB 実験 ↔ `paper_logs`（同一 dir で X/Y、`interv` で絞り込み） | 切替後全歩 / 切替前全歩 / 切替後N歩の Δx vs Y（`overall` / `α=1` 切替）・w_act は色+マーカー・w_act ごとの相関係数 r を UI・`*_corr.csv` に保存（図内には載せず）・`ready` のみ描画 |

CLI の `--group 'label:dir1,dir2'` と同じ考え方です。埋め込み描画と PNG 保存に対応。
eval/paper 一覧の dir 名末尾に、対応する学習 run の DB メモ（`| ...`）を付けて表示。
Δx vs 成功は `plot_deltax_success.py`（paper のみ、`--intervention-mode`、`--y-metric overall|alpha_1`、`--window`、`--post-n` 等）。軸ラベルは `Mean delta_x` / `overall success`（または `alpha=1 success`）。評価完了時に DB `softness_eval_runs` へ output_dir を記録。
Softness成功はスタイル切替（mean±std / 学習シード別細線）あり。CLI は `--style mean_std|per_seed`。

## データの保存場所

```
~/vnoid-experiments/
  runs/
    experiments.db          # experiments + softness_eval_runs（評価出力の紐付け）
    <run_id>/
      checkpoint/
      training_stats.csv
      result.json
      train.log
  evals/
    _logs/
    <runId>_s<min>-<max>_step<step>/
      results.csv
      summary.csv
  paper_logs/
    <runId>_s.../
      seed*_softness_*/control.csv
  plots/
```

## 起動方法

```bash
micromamba activate robot_env
python -m rl_experiment_app.main
```

## CLI からも従来どおり可能

```bash
cd python_scripts
python train_humanoid.py --run-id my_exp --w-track 1.2 --terrain soft --num-iterations 50
python eval_softness_grid.py --checkpoint-dir ~/vnoid-experiments/runs/my_exp/checkpoint \
  --softness-min 0.0 --softness-max 1.2 --softness-step 0.1 \
  --intervention-mode after_switch
python record_humanoid.py --checkpoint-dir ~/vnoid-experiments/runs/my_exp/checkpoint \
  --run-dir ~/vnoid-experiments/runs/my_exp --terrain soft
```

介入モード（`--intervention-mode` / History の interv）:

- `none` … 常に zeros（旧 `--no-rl-policy`）
- `full` … 常に学習済み方策（既定）
- `after_switch` … 地面切替後のみ方策

Sink vs Δx（`plot_sink_vs_deltax.py` / Plot タブ）は **contact 列付きの新規 paper_log** が必要です。
軟弱路面適用後・行動出力の1制御周期前の接地足 sink を横軸、そのときの `delta_x` を縦軸にします。

## 地盤（terrain）の扱い

エピソードは**必ず硬地盤から始まり**、C++側が決めたタイミングで地盤が切り替わります。
`--terrain` で指定できるのは**切り替え先**です。
