"""
Choreonoid Gymnasium 環境
=========================
VnoidRLController (Choreonoid 側 C++) を Gymnasium の標準インターフェースで
ラップした End-to-End 強化学習環境。

観測空間 (71 次元, float64):
  [ 0- 2]  base_pos    : ルートリンク位置 (x, y, z)  [m]
  [ 3- 6]  ori_quat    : ルートリンク姿勢クォータニオン (w, x, y, z)
  [ 7- 9]  base_vel    : ルートリンク線速度 (x, y, z)  [m/s]
  [10-12]  base_angvel : ルートリンク角速度 (x, y, z)  [rad/s]
  [13-41]  joint_q     : 全関節角度 (29 joints)  [rad]
  [42-70]  joint_dq    : 全関節角速度 (29 joints)  [rad/s]

行動空間 (29 次元, float32, [-1, 1]):
  各関節の目標角度。C++ 側で関節可動域にスケールされて PD 制御に使われる。

起動手順:
  1. Python でこのモジュールをインポートし env.reset() を呼ぶ
     (reset() の初回呼び出しで共有メモリが作成される)
  2. 別ターミナルで Choreonoid を起動する:
       ENV_ID=0 choreonoid <project_file>
  3. env.step(action) で強化学習ループを回す
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple
from shm_interface import ChoreonoidShmClient, NUM_JOINTS, NUM_OBSERVATIONS


class ChoreonoidEnv(gym.Env):
    """
    Choreonoid End-to-End 強化学習環境 (Gymnasium 準拠)。

    モジュール docstring に記載した観測空間・行動空間を持つ。
    並列実行時は env_id ごとに別プロセスの Choreonoid を起動すること。

    Args:
        env_id:            環境 ID。Choreonoid の ENV_ID 環境変数と対応する。
        max_episode_steps: 1 エピソードの最大ステップ数。超えると truncated=True になる。
    """

    metadata = {
        "render_modes": [],
        "render_fps": 60
    }

    def __init__(self, env_id: int = 0, max_episode_steps: int = 1000):
        super().__init__()

        self.env_id            = env_id
        self.max_episode_steps = max_episode_steps
        self._step_count       = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_OBSERVATIONS,),
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(NUM_JOINTS,),
            dtype=np.float32,
        )

        # 初回 reset() まで共有メモリは作成しない
        self.shm_client: Optional[ChoreonoidShmClient] = None

        print(f"[ChoreonoidEnv] 初期化完了 (env_id={env_id})")
        print(f"  観測空間: {self.observation_space.shape}")
        print(f"  行動空間: {self.action_space.shape}")

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        環境をリセットして初期観測を返す。

        初回呼び出し時に共有メモリを作成する。その後、Choreonoid を
        ENV_ID=<env_id> で起動すること (Choreonoid が共有メモリに接続する)。

        Args:
            seed:    乱数シード (Gymnasium 準拠; 現在はシミュレータに未伝達)
            options: 追加オプション (現在未使用)

        Returns:
            obs:  初期観測ベクトル, shape=(71,), dtype=float64
            info: 追加情報 (空 dict)
        """
        super().reset(seed=seed)

        if self.shm_client is None:
            self.shm_client = ChoreonoidShmClient(self.env_id)
            print(f"[Env {self.env_id}] 共有メモリ作成完了。Choreonoid を起動してください:")
            print(f"  ENV_ID={self.env_id} choreonoid <project_file>")

        obs = self.shm_client.reset(timeout=30.0)
        self._step_count = 0

        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        行動を送信して 1 ステップ進める。

        Args:
            action: 行動ベクトル, shape=(29,)。[-1, 1] の範囲にクリップされる。

        Returns:
            obs:       観測ベクトル, shape=(71,), dtype=float64
            reward:    ステップ報酬 (float)
            terminated: C++ 側の終了条件 (転倒など) が成立した場合 True
            truncated:  max_episode_steps に達した場合 True
            info:       追加情報 (空 dict)
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        obs, reward, terminated, info = self.shm_client.step(action, timeout=10.0)

        self._step_count += 1
        truncated = (self._step_count >= self.max_episode_steps) and (not terminated)

        return obs, reward, terminated, truncated, info

    def close(self):
        """共有メモリを解放して環境をクローズする。"""
        if self.shm_client is not None:
            self.shm_client.close()
            self.shm_client = None


# ---------------------------------------------------------------------------
# Ray RLlib 用ファクトリ関数
# ---------------------------------------------------------------------------

def make_choreonoid_env(config: dict) -> ChoreonoidEnv:
    """
    Ray RLlib の EnvContext から ChoreonoidEnv を生成するファクトリ関数。

    env_id はワーカー・環境ごとに一意になるよう以下の式で計算する:
        env_id = worker_index * num_envs_per_worker + vector_index

    worker_index + vector_index の単純な加算では、例えば
        worker=1, vector=0 → 1
        worker=0, vector=1 → 1  (衝突！)
    となるため、掛け算で一意性を保証する。

    Args:
        config: RLlib の EnvContext (worker_index, vector_index,
                num_envs_per_env_runner, max_episode_steps, etc.)

    Returns:
        ChoreonoidEnv インスタンス
    """
    worker_index      = config.get("worker_index", 0)
    vector_index      = config.get("vector_index", 0)
    num_envs_per_worker = config.get("num_envs_per_env_runner", 1)

    env_id = worker_index * num_envs_per_worker + vector_index
    return ChoreonoidEnv(
        env_id=env_id,
        max_episode_steps=config.get("max_episode_steps", 1000),
    )


# ---------------------------------------------------------------------------
# 簡易動作確認
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Choreonoid 環境テスト")
    print("=" * 70)

    env = ChoreonoidEnv(env_id=0)

    print("\nリセット中 (先に Choreonoid を起動してください)...")
    obs, _ = env.reset()
    print(f"観測: shape={obs.shape}, mean={obs.mean():.3f}, std={obs.std():.3f}")

    print("\nランダム行動で 10 ステップ実行...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(
            f"  Step {i + 1:2d}: reward={reward:+.3f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            print("  → エピソード終了")
            break

    env.close()
    print("\n完了")
