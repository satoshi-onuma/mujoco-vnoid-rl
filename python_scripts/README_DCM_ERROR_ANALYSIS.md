# DCM誤差分析ツール

ベース部のローカル座標系でDCM（Divergent Component of Motion）の誤差を分析するツールです。

## 概要

このツールは以下の機能を提供します：

1. **CSVデータ拡張**: ベース部の位置・姿勢（クォータニオン）を`control_log.csv`に記録
2. **座標変換**: グローバル座標のDCMをベース部ローカル座標系に変換
3. **誤差計算**: 参照値と実測値の差分を計算
4. **可視化**: 時系列グラフ、軌跡プロット、誤差分布などを生成
5. **データ出力**: 分析結果をCSVとPDFで出力

## ファイル構成

```
controller/vnoid_rl_env/bindings.cpp  # CSVログ機能（拡張済み）
python_scripts/analyze_dcm_error.py   # DCM誤差分析スクリプト
```

## 使用方法

### 1. データ収集

レンダリングを有効にしてシミュレーションを実行すると、`control_log.csv`が自動生成されます。

```bash
# 例: レンダリング有効で実行
python python_scripts/record_humanoid.py
```

### 2. DCM誤差分析

生成された`control_log.csv`を使って誤差分析を実行します。

```bash
cd /home/satoshi/vnoid-mujoco
python python_scripts/analyze_dcm_error.py control_log.csv
```

### 3. 出力ファイル

以下のファイルが生成されます：

- `dcm_error_analysis.csv`: 誤差データ（数値）
- `dcm_error_analysis.pdf`: グラフ（4ページ）

## 出力グラフの内容

### ページ1: DCM誤差時系列
- X方向（前後）誤差
- Y方向（左右）誤差
- Z方向（上下）誤差

### ページ2: 誤差ノルムと接触状態
- DCM誤差の大きさ（ノルム）の時系列
- 足の接触状態（右足/左足）

### ページ3: DCM軌跡（XY平面）
- ベース部から見たDCMの動き
- 参照軌跡 vs 実測軌跡

### ページ4: DCM軌跡（XZ・YZ平面）
- XZ平面（前後・上下）
- YZ平面（左右・上下）

## 座標変換の詳細

### ベース部ローカル座標系への変換式

グローバル座標のDCMをベース部ローカル座標系に変換する手順：

1. **平行移動**: `relative_pos = dcm_global - base_pos`
2. **回転変換**: `dcm_local = base_ori.conjugate() * relative_pos`

### クォータニオンによる回転

クォータニオン `q = [w, x, y, z]` でベクトル `v` を回転：

```
v_rotated = q * v * q^*
```

ここで `q^*` はクォータニオンの共役（`[w, -x, -y, -z]`）です。

## CSVフィールド（追加分）

`control_log.csv`に以下のフィールドが追加されています：

- `base_pos_x`, `base_pos_y`, `base_pos_z`: ベース部実際位置
- `base_pos_ref_x`, `base_pos_ref_y`, `base_pos_ref_z`: ベース部目標位置
- `base_ori_w`, `base_ori_x`, `base_ori_y`, `base_ori_z`: ベース部実際姿勢
- `base_ori_ref_w`, `base_ori_ref_x`, `base_ori_ref_y`, `base_ori_ref_z`: ベース部目標姿勢

## 誤差評価指標

スクリプトは以下の統計量を計算します：

- **平均誤差**: 各軸の平均偏差
- **RMS誤差**: 二乗平均平方根誤差
- **最大誤差**: 誤差ノルムの最大値

## トラブルシューティング

### ファイルが見つからない

```
FileNotFoundError: control_log.csv
```

→ レンダリングを有効にしてシミュレーションを実行してください。

### 必要なカラムがない

```
ValueError: 必要なカラムがありません
```

→ `bindings.cpp`が最新版にコンパイルされているか確認してください。

```bash
cd /home/satoshi/vnoid-mujoco
mkdir -p build && cd build
cmake ..
make
```

## 依存パッケージ

- `numpy`: 数値計算
- `pandas`: データ処理
- `matplotlib`: グラフ描画

インストール：
```bash
pip install numpy pandas matplotlib
```

## 参考文献

DCM制御の理論については以下を参照：

- Englsberger, J., et al. (2015). "Three-dimensional bipedal walking control based on Divergent Component of Motion"
- 本プロジェクトの制御アルゴリズム: `src/stabilizer.cpp`

## ライセンス

本プロジェクトと同じライセンスに従います。
