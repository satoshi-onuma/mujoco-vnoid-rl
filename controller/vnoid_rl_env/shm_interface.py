"""
共有メモリインターフェース (Python 側)
======================================
Choreonoid C++ コントローラ (VnoidRLController) との POSIX 共有メモリ通信を担う。

共有メモリレイアウト (824 bytes):
  offset   0  ControlBlock (24 bytes)  ← 制御フラグ・報酬
  offset  24  ActionBlock  (232 bytes) ← Python → C++: 関節目標角度
  offset 256  StateBlock   (568 bytes) ← C++ → Python: 観測ベクトル

同期プロトコル (step の場合):
  1. Python: ready = 0 をクリア
  2. Python: action を書き込み
  3. Python: step_request = 1 をセット
  4. C++:    step_request を検知 → 制御実行
  5. C++:    obs / reward / done を書き込み
  6. C++:    ready = 1 をセット (メモリバリア後)
  7. Python: ready == 1 をポーリングで確認
  8. Python: obs / reward / done を読み取り

手順 1 (ready=0 クリア) を省略すると、前回の ready=1 が残っていて
C++ の処理完了前に Python が即リターンしてしまうため必須。
"""

import mmap
import os
import struct
import time
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# 次元数定数 (C++ の shm_interface.h と一致させること)
# ---------------------------------------------------------------------------

NUM_JOINTS       = 29
NUM_OBSERVATIONS = 71

# ---------------------------------------------------------------------------
# メモリレイアウト定数
# ---------------------------------------------------------------------------

# ControlBlock: int × 6 = 24 bytes  (_pad で 8 byte アライメント調整済み)
# ActionBlock:  double × 29 = 232 bytes   offset: 24
# StateBlock:   double × 71 = 568 bytes   offset: 256
CONTROL_BLOCK_SIZE = 4 * 6                 # 24 bytes
ACTION_BLOCK_SIZE  = 8 * NUM_JOINTS        # 232 bytes
STATE_BLOCK_SIZE   = 8 * NUM_OBSERVATIONS  # 568 bytes
TOTAL_SIZE         = CONTROL_BLOCK_SIZE + ACTION_BLOCK_SIZE + STATE_BLOCK_SIZE  # 824 bytes

OFFSET_CONTROL = 0
OFFSET_ACTION  = CONTROL_BLOCK_SIZE                         # 24
OFFSET_STATE   = CONTROL_BLOCK_SIZE + ACTION_BLOCK_SIZE     # 256

# ---------------------------------------------------------------------------
# ControlBlock フィールドのインデックス (int 単位、各 4 bytes)
# ---------------------------------------------------------------------------
# C++ 側の ControlBlock メモリ上の並び:
#   byte  0- 3 : step_request   (Python → C++)
#   byte  4- 7 : reset_request  (Python → C++)
#   byte  8-11 : done           (C++ → Python)
#   byte 12-15 : ready          (C++ → Python)
#   byte 16-19 : reward_bits    (C++ → Python, float の IEEE 754 ビット表現)
#   byte 20-23 : _pad           (未使用)

_IDX_STEP_REQUEST  = 0  # byte offset  0
_IDX_RESET_REQUEST = 1  # byte offset  4
_IDX_DONE          = 2  # byte offset  8
_IDX_READY         = 3  # byte offset 12
_IDX_REWARD_BITS   = 4  # byte offset 16


# ---------------------------------------------------------------------------
# ChoreonoidShmClient
# ---------------------------------------------------------------------------

class ChoreonoidShmClient:
    """
    Choreonoid 共有メモリクライアント。

    Python (RL エージェント側) が共有メモリを *作成* し、
    Choreonoid (C++ 側) がそれに *接続* する想定で動作する。
    したがって、このクライアントを生成 *してから* Choreonoid を起動すること。

    Args:
        env_id: 環境 ID。並列実行時はワーカーごとに異なる値を使う。
                Choreonoid 起動時に ENV_ID=<env_id> を環境変数で渡す。
    """

    def __init__(self, env_id: int = 0):
        self.env_id   = env_id
        self.shm_name = f"/vnoid_rl_shm_{env_id}"
        self.shm_mmap = None
        self._create_shared_memory()

    def _create_shared_memory(self):
        """共有メモリを作成してゼロ初期化する。"""
        import posix_ipc

        # 残存する同名の共有メモリがあれば先に削除する
        try:
            posix_ipc.unlink_shared_memory(self.shm_name)
        except posix_ipc.ExistentialError:
            pass

        self.shm = posix_ipc.SharedMemory(
            self.shm_name,
            flags=posix_ipc.O_CREAT,
            mode=0o666,
            size=TOTAL_SIZE,
        )

        self.shm_mmap = mmap.mmap(self.shm.fd, TOTAL_SIZE)
        os.close(self.shm.fd)  # mmap 後は fd 不要

        self.shm_mmap.seek(0)
        self.shm_mmap.write(b'\x00' * TOTAL_SIZE)

        print(f"[ShmClient] 共有メモリ作成完了: {self.shm_name} ({TOTAL_SIZE} bytes)")

    # -----------------------------------------------------------------------
    # 公開 API
    #__から始まるのは非公開であることを暗黙的に含んでいるらしい
    # -----------------------------------------------------------------------

    def reset(self, timeout: float = 10.0) -> np.ndarray:
        """
        環境をリセットしてリセット後の観測を返す。

        同期プロトコル:
          1. ready = 0 をクリア  (前回の値が残らないよう)
          2. reset_request = 1 をセット
          3. ready == 1 になるまでポーリング
          4. reset_request = 0 をクリア
          5. 観測を読み取って返す

        Args:
            timeout: タイムアウト秒数。Choreonoid が応答しない場合に例外を投げる。

        Returns:
            observation: 観測ベクトル, shape=(71,), dtype=float64
        """
        self._write_control_int(_IDX_READY, 0)          # ready = 0
        self._write_control_int(_IDX_RESET_REQUEST, 1)  # reset_request = 1

        self._wait_ready(timeout)

        self._write_control_int(_IDX_RESET_REQUEST, 0)  # reset_request = 0 (クリア)

        return self._read_state()

    def step(
        self,
        action: np.ndarray,
        timeout: float = 10.0,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        1 ステップ実行して結果を返す。

        同期プロトコル:
          1. ready = 0 をクリア
          2. action を ActionBlock に書き込み
          3. step_request = 1 をセット
          4. ready == 1 になるまでポーリング
          5. step_request = 0 をクリア
          6. obs / reward / done を読み取って返す

        Args:
            action:  行動ベクトル, shape=(29,), 正規化済み [-1, 1]
            timeout: タイムアウト秒数

        Returns:
            observation: shape=(71,), dtype=float64
            reward:      float
            done:        エピソード終了フラグ
            info:        追加情報 (現在は空 dict)
        """
        self._write_control_int(_IDX_READY, 0)  # ready = 0
        self._write_action(action)
        self._write_control_int(_IDX_STEP_REQUEST, 1)   # step_request = 1

        self._wait_ready(timeout)

        self._write_control_int(_IDX_STEP_REQUEST, 0)   # step_request = 0 (クリア)

        obs    = self._read_state()
        reward = self._read_reward()
        done   = self._read_control_int(_IDX_DONE) == 1

        return obs, reward, done, {}

    def close(self):
        """共有メモリを解放・削除する。"""
        if self.shm_mmap is not None:
            self.shm_mmap.close()
        if hasattr(self, 'shm'):
            
            self.shm.unlink()
        print(f"[ShmClient] 共有メモリ解放完了: {self.shm_name}")

    # -----------------------------------------------------------------------
    # 内部ヘルパー
    # -----------------------------------------------------------------------

    def _wait_ready(self, timeout: float):
        """C++ が ready=1 をセットするまでスピンウェイトする。"""
        deadline = time.time() + timeout
        while self._read_control_int(_IDX_READY) != 1:
            if time.time() > deadline:
                raise TimeoutError(
                    f"[ShmClient] C++ controller did not respond within {timeout}s"
                )
            time.sleep(0.001)

    def _write_control_int(self, index: int, value: int):
        """ControlBlock の index 番目の int フィールドに value を書き込む。"""
        self.shm_mmap.seek(OFFSET_CONTROL + index * 4)
        self.shm_mmap.write(struct.pack('i', value))

    def _read_control_int(self, index: int) -> int:
        """ControlBlock の index 番目の int フィールドを読み取る。"""
        self.shm_mmap.seek(OFFSET_CONTROL + index * 4)
        return struct.unpack('i', self.shm_mmap.read(4))[0]

    def _read_reward(self) -> float:
        """
        reward_bits を float に変換して返す。

        C++ 側は float を memcpy で int に変換してアトミックに書き込む。
        Python 側は 'f' フォーマット (IEEE 754, 4 bytes) でそのまま解釈する。
        """
        self.shm_mmap.seek(OFFSET_CONTROL + _IDX_REWARD_BITS * 4)
        return struct.unpack('f', self.shm_mmap.read(4))[0]

    def _write_action(self, action: np.ndarray):
        """行動ベクトルを ActionBlock に書き込む。"""
        if action.shape != (NUM_JOINTS,):
            raise ValueError(f"Action shape must be ({NUM_JOINTS},), got {action.shape}")
        self.shm_mmap.seek(OFFSET_ACTION)
        self.shm_mmap.write(action.astype(np.float64).tobytes())

    def _read_state(self) -> np.ndarray:
        """
        StateBlock から観測ベクトルを読み取る。

        frombuffer() が返すビューは読み取り専用なため .copy() で可変配列にする。
        """
        self.shm_mmap.seek(OFFSET_STATE)
        data = self.shm_mmap.read(STATE_BLOCK_SIZE)
        obs  = np.frombuffer(data, dtype=np.float64).copy()
        assert obs.shape == (NUM_OBSERVATIONS,), f"Observation shape mismatch: {obs.shape}"
        return obs


# ---------------------------------------------------------------------------
# 簡易動作確認
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("共有メモリインターフェース 簡易テスト")
    print("=" * 60)

    client = ChoreonoidShmClient(env_id=0)
    print(f"共有メモリ作成完了 ({TOTAL_SIZE} bytes)")
    print("次のコマンドで Choreonoid を起動してください:")
    print("  ENV_ID=0 choreonoid <project_file>")

    client.close()
