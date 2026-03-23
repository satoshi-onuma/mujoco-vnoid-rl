"""
共有メモリインターフェース（Python側）
Choreonoid C++コントローラとプロセス間通信を行う
"""

import mmap
import os
import struct
import time
import numpy as np
from typing import Tuple, Optional

# 定数定義（C++側と一致させる）
NUM_JOINTS = 29
NUM_OBSERVATIONS = 73

# 構造体サイズ計算
CONTROL_BLOCK_SIZE = 4 * 5  # int×4 + float×1 = 20 bytes
ACTION_BLOCK_SIZE = 8 * NUM_JOINTS  # double×29 = 232 bytes
STATE_BLOCK_SIZE = 8 * NUM_OBSERVATIONS  # double×73 = 584 bytes
TOTAL_SIZE = CONTROL_BLOCK_SIZE + ACTION_BLOCK_SIZE + STATE_BLOCK_SIZE  # 836 bytes


class ChoreonoidShmClient:
    """Choreonoid共有メモリクライアント"""
    
    def __init__(self, env_id: int = 0):
        self.env_id = env_id
        self.shm_name = f"/vnoid_rl_shm_{env_id}"
        self.shm_fd = None
        self.shm_mmap = None
        
        # オフセット定義
        self.offset_control = 0
        self.offset_action = CONTROL_BLOCK_SIZE
        self.offset_state = CONTROL_BLOCK_SIZE + ACTION_BLOCK_SIZE
        
        self._create_shared_memory()
    
    def _create_shared_memory(self):
        """共有メモリを作成"""
        import posix_ipc
        
        # 既存の共有メモリを削除（クリーンアップ）
        try:
            posix_ipc.unlink_shared_memory(self.shm_name)
        except posix_ipc.ExistentialError:
            pass
        
        # 共有メモリを作成
        self.shm = posix_ipc.SharedMemory(
            self.shm_name,
            flags=posix_ipc.O_CREAT,
            mode=0o666,
            size=TOTAL_SIZE
        )
        
        # メモリマップ
        self.shm_mmap = mmap.mmap(self.shm.fd, TOTAL_SIZE)
        os.close(self.shm.fd)  # fdは不要
        
        # 初期化
        self.shm_mmap.seek(0)
        self.shm_mmap.write(b'\x00' * TOTAL_SIZE)
        
        print(f"✅ 共有メモリ作成完了: {self.shm_name} ({TOTAL_SIZE} bytes)")
    
    def reset(self, timeout: float = 10.0) -> np.ndarray:
        """
        環境をリセット
        
        Returns:
            observation: 観測ベクトル (73,)
        """
        # reset_requestフラグを立てる
        self._write_control_int(1, 1)  # reset_request = 1
        
        # readyフラグが立つまで待機
        start_time = time.time()
        while True:
            if self._read_control_int(3) == 1:  # ready == 1
                break
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Reset timeout after {timeout}s")
            time.sleep(0.001)
        
        # reset_requestをクリア
        self._write_control_int(1, 0)
        
        # 観測を取得
        obs = self._read_state()
        
        return obs
    
    def step(self, action: np.ndarray, timeout: float = 10.0) -> Tuple[np.ndarray, float, bool, dict]:
        """
        1ステップ実行
        
        Args:
            action: 行動ベクトル (29,) 正規化済み [-1, 1]
            
        Returns:
            observation: 観測ベクトル (73,)
            reward: 報酬
            done: エピソード終了フラグ
            info: 追加情報
        """
        # 行動を書き込み
        self._write_action(action)
        
        # step_requestフラグを立てる
        self._write_control_int(0, 1)  # step_request = 1
        
        # readyフラグが立つまで待機（Choreonoid側の処理完了を待つ）
        start_time = time.time()
        while True:
            if self._read_control_int(3) == 1:  # ready == 1
                break
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Step timeout after {timeout}s")
            time.sleep(0.001)
        
        # step_requestをクリア
        self._write_control_int(0, 0)
        
        # 結果を取得
        obs = self._read_state()
        reward = self._read_control_float(4)  # reward
        done = self._read_control_int(2) == 1  # done
        
        info = {}
        
        return obs, reward, done, info
    
    def _write_control_int(self, index: int, value: int):
        """制御ブロックのint値を書き込み"""
        offset = self.offset_control + index * 4
        self.shm_mmap.seek(offset)
        self.shm_mmap.write(struct.pack('i', value))
    
    def _read_control_int(self, index: int) -> int:
        """制御ブロックのint値を読み取り"""
        offset = self.offset_control + index * 4
        self.shm_mmap.seek(offset)
        return struct.unpack('i', self.shm_mmap.read(4))[0]
    
    def _read_control_float(self, index: int) -> float:
        """制御ブロックのfloat値を読み取り（rewardのみ）"""
        offset = self.offset_control + 16  # int×4の後
        self.shm_mmap.seek(offset)
        return struct.unpack('f', self.shm_mmap.read(4))[0]
    
    def _write_action(self, action: np.ndarray):
        """行動ベクトルを書き込み"""
        assert action.shape == (NUM_JOINTS,), f"Action shape must be ({NUM_JOINTS},), got {action.shape}"
        self.shm_mmap.seek(self.offset_action)
        self.shm_mmap.write(action.astype(np.float64).tobytes())
    
    def _read_state(self) -> np.ndarray:
        """観測ベクトルを読み取り"""
        self.shm_mmap.seek(self.offset_state)
        data = self.shm_mmap.read(STATE_BLOCK_SIZE)
        obs = np.frombuffer(data, dtype=np.float64)
        assert obs.shape == (NUM_OBSERVATIONS,), f"Observation shape mismatch: {obs.shape}"
        return obs
    
    def close(self):
        """共有メモリをクリーンアップ"""
        if self.shm_mmap is not None:
            self.shm_mmap.close()
        if hasattr(self, 'shm'):
            self.shm.close_fd()
            self.shm.unlink()
        print(f"✅ 共有メモリクリーンアップ完了: {self.shm_name}")


if __name__ == "__main__":
    # 簡易テスト
    print("共有メモリインターフェーステスト")
    client = ChoreonoidShmClient(env_id=0)
    print("共有メモリ作成完了")
    print("Choreonoidを起動してください: ENV_ID=0 choreonoid ...")
    client.close()
