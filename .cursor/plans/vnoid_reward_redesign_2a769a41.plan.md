---
name: vnoid reward redesign
overview: 行動空間（foot_offset 2次元）は変えず、vnoid_rl_env の報酬を「速度追従報酬 + 弱い出力ペナルティ」に再設計する。前提として現状コメントアウトされている歩行（stepping_controller）を有効化し、joystick 相当の stride/sway 指令を目標速度として扱う。
todos:
  - id: enable-walking
    content: "controller/vnoid_rl_env/myrobot.cpp: footstep再計画ブロック(L170-188)とstepping_controller.Update(L190)を有効化し、固定strideで定常前進歩行を成立させる（ユーザーが有効化済み）"
    status: completed
  - id: command-func
    content: 指令を関数化・メンバ化。WalkCommand構造体(src/types.h) + MyRobot::walk_cmd(真実源) + UpdateFootstepPlan()。指令設定はフラグ分岐でなく用途別の別関数 SetFixedWalkCommand()/SetWalkCommand(stride,sway,turn) に分ける。bindings に set_walk_command を公開
    status: completed
  - id: reward-state
    content: "bindings.cpp: previous_y等のメンバ追加とreset初期化（L221-228, L343周辺）"
    status: completed
  - id: reward-encapsulate
    content: compute_reward() を項ごとの関数に分割してカプセル化（reward_tracking/reward_action_penalty/reward_healthy）。重み(w_track/tracking_sigma/w_act/w_healthy)はメンバ化。compute_reward は合算のみ。各項を個別ログ可能に
    status: completed
  - id: tracking-reward
    content: "reward_tracking(): 1歩あたり変位をbaseフレームで評価し walk_cmd を目標に exp(-err/sigma)。既存forward_reward*5.0を置換"
    status: completed
  - id: action-penalty
    content: "reward_action_penalty(): last_rl_action を用いた ||a||^2 を実装（合成時に w_act で弱く減算）"
    status: completed
  - id: cmd-randomize
    content: role_model準拠でstride/sway/turnをランダム化し set_walk_command で注入。観測へ指令追加(obs次元拡張, my_humanoid_env.py の observation_space も更新)。報酬目標は walk_cmd を読むので変更不要
    status: completed
  - id: tracking-rotation-fix
    content: reward_tracking() の回転基準を修正。base.oriのフル姿勢・歩終了時点ではなく、base.angle.z()（胴体yaw=両足の中間角、歩開始時点）でyaw成分のみ2D回転する。支持脚(foot_ori[sup])基準は旋回時にturn/2相当のバイアスが乗るため不採用。compute_reward()末尾でstep_start_yawを次歩用に保存し、reset()でも初期化する
    status: completed
isProject: false
---

# vnoid RL 報酬再設計（速度追従 + 出力ペナルティ）

## 背景と方針
- 行動空間は現状維持: `action=[foot_offset_x, foot_offset_y]`（[python_scripts/my_humanoid_env.py](python_scripts/my_humanoid_env.py) L65, L83-85）。変更は「歩行の有効化」と「報酬」のみ。
- 目標速度は role_model（[role_model/sbr1_env.py](role_model/sbr1_env.py) L271-274 の `tracking_lin_vel`）に倣い、`exp(-error^2/sigma)` 型で定義。
- 「指令 = footstep の stride/sway」と解釈する（[controller/sample_controller/myrobot.cpp](controller/sample_controller/myrobot.cpp) L189-200 の joystick→stride/sway/turn マッピングと同型）。
- 指令はベタ書きせず関数化・メンバ化し、固定指令→ランダム指令へ差し替え可能にする（Phase 1.5）。報酬側は同じメンバを目標として読む。
- 報酬は `step()` 内で 1歩に1回だけ算出される（[controller/vnoid_rl_env/bindings.cpp](controller/vnoid_rl_env/bindings.cpp) L860-888）ため、瞬時速度でなく「1歩あたり変位」で比較する。

## 現状の問題（先に潰す）
- [controller/vnoid_rl_env/myrobot.cpp](controller/vnoid_rl_env/myrobot.cpp) L190 の `stepping_controller.Update(...)` がコメントアウト → ロボットは前進せず、`foot_offset` も stepping_controller 内（[src/stepping_controller.cpp](src/stepping_controller.cpp) L128-129）で適用されるので事実上無効。
- 初期 footstep が `stride=0`（[controller/vnoid_rl_env/myrobot.cpp](controller/vnoid_rl_env/myrobot.cpp) L140-141）。
- したがって歩行を有効化しないと速度追従も出力ペナルティも意味を持たない。

## Phase 1: 歩行の有効化（完了・ユーザー実施済み）
対象: [controller/vnoid_rl_env/myrobot.cpp](controller/vnoid_rl_env/myrobot.cpp)
- L170-188 の footstep 再計画ブロックが有効化済み（10制御周期ごとに `stride=0.1, spacing=0.2, duration=0.4` で4歩 push → `Plan`/`GenerateDCM`）。
- L190 の `stepping_controller.Update(..., rl_params)` 有効化済み。`foot_offset` が [src/stepping_controller.cpp](src/stepping_controller.cpp) L128-129 に流れる。
- 残確認: 「stride による定常前進 + RLバランス補正」が転ばず成立するか描画/CSVで確認。

## Phase 1.5: 指令の関数化（ランダム化への布石）
現状 [controller/vnoid_rl_env/myrobot.cpp](controller/vnoid_rl_env/myrobot.cpp) L175-184 で指令がベタ書き。これを抽象化し、固定→ランダムを差し替え可能にする。報酬側の目標値も同じメンバから読む。

1. 指令の型（[src/types.h](src/types.h) の `RLParams`(L18-20) の近く）
```cpp
struct WalkCommand {
    double stride   = 0.0;
    double sway     = 0.0;
    double turn     = 0.0;
    double spacing  = 0.2;
    double climb    = 0.0;
    double duration = 0.4;
};
```

2. `MyRobot` にメンバ + 適用メソッド（[controller/vnoid_rl_env/myrobot.h](controller/vnoid_rl_env/myrobot.h), [controller/vnoid_rl_env/myrobot.cpp](controller/vnoid_rl_env/myrobot.cpp)）
   - `WalkCommand walk_cmd;` をメンバ化。**これを唯一の真実源**とし、reward もこれを読む。
   - `void UpdateFootstepPlan();` … L170-188 の再計画ロジックを移動し、`walk_cmd` を使って `step` を構築。指令の中身は一切ここで決めない（`walk_cmd` を読むだけ）。
   - `Control()` 内は再計画条件（`timer.count % (10*control_cycle)==0`）のまま `UpdateFootstepPlan();` を呼ぶだけに置換。指令生成は行わない。

3. 指令の設定は「フラグで優先度分岐」ではなく**用途別の別関数**にする（デバッグしやすく意図が明確）
   - `void SetFixedWalkCommand();` … 固定前進の既定値（`stride=0.1, spacing=0.2, duration=0.4` 等）を `walk_cmd` に代入。`Init()`/reset で呼ぶ現行相当の運用。
   - `void SetWalkCommand(double stride, double sway, double turn);` … 外部（学習側）から `walk_cmd` を上書き。ランダム化運用で使う。
   - どちらも「`walk_cmd` を書くだけ」で、`UpdateFootstepPlan()`/reward は常に最新の `walk_cmd` を読む。優先フラグや分岐は不要（最後に呼んだ関数が勝つ、という単純規則）。
   - [controller/vnoid_rl_env/bindings.cpp](controller/vnoid_rl_env/bindings.cpp) に pybind ラッパ `set_walk_command(...)` を追加 → Python の `reset()`/リサンプルから注入（role_model `_resample_commands` [role_model/sbr1_env.py](role_model/sbr1_env.py) L131-134 相当）。固定運用時は呼ばなければ `SetFixedWalkCommand()` の値のまま。

## Phase 2〜3: 報酬カプセル化・指令ランダム化（実装済み）
[controller/vnoid_rl_env/bindings.cpp](controller/vnoid_rl_env/bindings.cpp) に実装済み:
- `previous_x/previous_y`、`w_track/tracking_sigma/w_act/w_healthy`、`reward_tracking()/reward_action_penalty()/reward_healthy()`、`compute_reward()`（合算のみ）、`set_walk_command()`（pybind公開）、`get_observation()`末尾への`walk_cmd`追加（obs 16→19次元）。
- [python_scripts/my_humanoid_env.py](python_scripts/my_humanoid_env.py): `observation_space`を19次元に、`reset()`でstride/sway/turnをランダムサンプルして`set_walk_command()`で注入。

## Phase 2 修正: reward_tracking() の回転基準の修正（今回対応）
現状の実装（下記）はロボット胴体の**歩終了時点**の姿勢で回転しており、[src/footstep_planner.cpp](src/footstep_planner.cpp) が定義する `stride` の基準（**支持脚の向き**、**歩開始時点**）とズレている。

```108:116:controller/vnoid_rl_env/bindings.cpp
    double reward_tracking() {
        if (!robot) return 0.0;
        const double dx = d->qpos[0] - previous_x;
        const double dy = d->qpos[1] - previous_y;
        const Vector3 disp_local = robot->base.ori.conjugate() * Vector3(dx, dy, 0.0);
        const double ex = robot->walk_cmd.stride - disp_local.x();
        const double ey = robot->walk_cmd.sway   - disp_local.y();
        return std::exp(-(ex*ex + ey*ey) / tracking_sigma);
    }
```

当初「支持脚の向き（`foot_ori[sup]`）」を基準にする案を検討したが、これはバイアスを持つため却下。根拠:
```144:149:src/stepping_controller.cpp
    double angle_diff = foot[1].angle_ref.z() - foot[0].angle_ref.z();
    while(angle_diff >  pi) angle_diff -= 2.0*pi;
    while(angle_diff < -pi) angle_diff += 2.0*pi;
	base.angle_ref.z() = foot[0].angle_ref.z() + angle_diff/2.0;
    base.ori_ref   = FromRollPitchYaw(base.angle_ref);
```
胴体（base）の向きは「両足の中間角」として明示的に定義されている。また[src/footstep_planner.cpp](src/footstep_planner.cpp) L57 `st1.foot_angle[swg] = st0.foot_angle[sup] + Vector3(0.0,0.0,dtheta)` により、旋回時は支持脚は古い向きを保持し遊脚だけが`turn`分丸ごと新しい向きになる。そのため「支持脚の向き」を基準にすると、体全体の実際の向き（両足の中間）から系統的に`turn/2`相当ズレたバイアスが乗る。

修正方針（決定済み）:
- 基準の「向き」は**ロボット胴体（base）の実測yaw** `robot->base.angle.z()`（[src/robot_base.h](src/robot_base.h) L50、両足の中間角に追従する実測値）を使う。特定の脚の向きではない。
- 回転はyaw成分のみ（roll/pitchは無視）。
- 基準タイミングは「歩開始時点」＝ちょうど`compute_reward()`が呼ばれる瞬間（支持脚切替直後、[controller/vnoid_rl_env/bindings.cpp](controller/vnoid_rl_env/bindings.cpp) L860-861の`step_completed`判定と同じ境界）。この対称性を使い、`compute_reward()`末尾で「次の歩用の基準yaw」を保存する。

実装:
1. メンバ追加: `double step_start_yaw = 0.0;`
2. `reward_tracking()`を2D回転（yawのみ）に変更:
```cpp
double reward_tracking() {
    if (!robot) return 0.0;
    const double dx = d->qpos[0] - previous_x;
    const double dy = d->qpos[1] - previous_y;
    const double c = std::cos(-step_start_yaw), s = std::sin(-step_start_yaw);
    const double ex_disp = c*dx - s*dy;
    const double ey_disp = s*dx + c*dy;
    const double ex = robot->walk_cmd.stride - ex_disp;
    const double ey = robot->walk_cmd.sway   - ey_disp;
    return std::exp(-(ex*ex + ey*ey) / tracking_sigma);
}
```
3. `compute_reward()`末尾（`previous_x/y`更新と同じ場所）で次歩用の基準yawを保存:
```cpp
previous_x = d->qpos[0];
previous_y = d->qpos[1];
step_start_yaw = robot->base.angle.z();
```
   - `robot->base`は`MyRobot`の公開メンバ（[controller/vnoid_rl_env/myrobot.h](controller/vnoid_rl_env/myrobot.h) L28）なので追加includeは不要。
4. `reset()`内でも`step_start_yaw`を初期化（`previous_x/y`初期化と同じ場所、L221-228相当）。

## 検証（ユーザー担当）
- Phase 1〜3（歩行有効化・関数化・カプセル化・ランダム化）は完了済み。
- 今回の回転基準修正をビルド・スモークテストまで実施し、以降の学習・チューニング（`w_act=0`で追従確認→弱く微増、転倒率確認）はユーザー側で実施。