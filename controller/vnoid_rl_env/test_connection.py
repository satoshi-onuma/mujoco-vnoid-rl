"""
共有メモリ疎通テスト
Choreonoid起動前にこのスクリプトを実行して、共有メモリが正常に動作するか確認
"""

import time
import numpy as np
from shm_interface import ChoreonoidShmClient, NUM_JOINTS, NUM_OBSERVATIONS

print("=" * 70)
print("共有メモリ疎通テスト")
print("=" * 70)

# 環境ID 0で共有メモリを作成
client = ChoreonoidShmClient(env_id=0)

print("\n✅ 共有メモリ作成完了")
print(f"   観測次元: {NUM_OBSERVATIONS}")
print(f"   行動次元: {NUM_JOINTS}")

print("\n📢 Choreonoidを別ターミナルで起動してください:")
print("   ENV_ID=0 ~/choreonoid/build/bin/choreonoid ~/choreonoid/ext/vnoid/project/vnoid_rl_project.cnoid --start-simulation")

print("\nChoreonoid起動待機中（30秒）...")
print("Choreonoid側がready=1を設定するまで待ちます")

# ready フラグを監視
start_time = time.time()
timeout = 30.0

while True:
    # ready フラグを確認
    import struct
    client.shm_mmap.seek(12)  # ready のオフセット
    ready = struct.unpack('i', client.shm_mmap.read(4))[0]
    
    if ready == 1:
        print("\n✅ Choreonoidから準備完了信号を受信！")
        break
    
    elapsed = time.time() - start_time
    if elapsed > timeout:
        print(f"\n❌ タイムアウト（{timeout}秒）")
        print("Choreonoidが起動していないか、共有メモリに接続できていません")
        client.close()
        exit(1)
    
    time.sleep(0.1)
    if int(elapsed) % 5 == 0 and elapsed > 0:
        print(f"  待機中... {int(elapsed)}秒経過")

print("\n" + "=" * 70)
print("疎通テスト成功！")
print("=" * 70)
print("\n次のステップ:")
print("1. python choreonoid_env.py  # 環境クラスのテスト")
print("2. python train_choreonoid.py  # 学習開始")

client.close()
