# setup_cluster.py
# Rayクラスタを起動し、Workerの接続を待つスクリプト

import ray
import time
import os

print("=" * 70)
print("🌐 Rayクラスタセットアップ")
print("=" * 70)

# 既存のRayをクリーンアップ
try:
    ray.shutdown()
except:
    pass

# Rayクラスタを起動
print("\n📡 Rayクラスタを起動中...")
context = ray.init(
    include_dashboard=True,
    dashboard_host='0.0.0.0',
    dashboard_port=8265,
    logging_level='ERROR',
    _temp_dir='/tmp/ray'
)

# IPアドレスを取得
import socket
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

my_ip = get_ip()

print("✅ Rayクラスタ起動完了\n")
print(f"📍 このPCのIPアドレス: {my_ip}")
print(f"🖥️  現在のノード数: {len(ray.nodes())}")
print(f"💻 利用可能CPU: {ray.cluster_resources().get('CPU', 0)}")
print(f"📊 ダッシュボード: http://{my_ip}:8265")

print("\n" + "=" * 70)
print("別PCからWorkerを接続するには:")
print("=" * 70)
print(f"1. 別PCで以下の環境変数を設定:")
print(f"   export RAY_ADDRESS='{my_ip}:6379'")
print(f"\n2. Pythonスクリプトで接続:")
print(f"   python connect_worker.py")
print("=" * 70)

# Workerの接続を待つ
print("\n⏳ Workerの接続を待機中...")
print("   - 1台のPCだけで学習する場合: そのままEnterキーを押してください")
print("   - 別PCを追加する場合: 接続完了後にEnterキーを押してください")
print()

initial_nodes = len(ray.nodes())
while True:
    current_nodes = len(ray.nodes())
    if current_nodes > initial_nodes:
        print(f"\n✅ 新しいノードが接続されました！ (合計: {current_nodes}ノード)")
        print(f"   利用可能CPU: {ray.cluster_resources().get('CPU', 0)}")
    
    # ユーザー入力をチェック
    import select
    import sys
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        input()
        break
    
    time.sleep(2)

print("\n" + "=" * 70)
print("クラスタ構成完了")
print("=" * 70)
print(f"🖥️  ノード数: {len(ray.nodes())}")
print(f"💻 総CPU数: {ray.cluster_resources().get('CPU', 0)}")
print(f"💾 総メモリ: {ray.cluster_resources().get('memory', 0) / (1024**3):.1f} GB")

# NUM_WORKERSの推奨値を計算
total_cpus = int(ray.cluster_resources().get('CPU', 0))
recommended_workers = max(1, total_cpus - 1)  # 1つはLearner用に残す

print(f"\n💡 推奨設定:")
print(f"   NUM_WORKERS = {recommended_workers}")

print("\n" + "=" * 70)
print("このターミナルは開いたままにして、")
print("新しいターミナルで学習スクリプトを実行してください:")
print("=" * 70)
print(f"  cd ~/vnoid-mujoco/python_scripts")
print(f"  python train_humanoid.py")
print(f"\nまたは、NUM_WORKERSを{recommended_workers}に設定してから:")
print(f"  python train_humanoid_integrated.py")
print("=" * 70)

# クラスタを維持
print("\n🔄 クラスタを起動したまま維持します...")
print("   Ctrl+C で終了\n")

try:
    while True:
        time.sleep(10)
        nodes = len(ray.nodes())
        cpus = ray.cluster_resources().get('CPU', 0)
        print(f"[{time.strftime('%H:%M:%S')}] ノード: {nodes}, CPU: {cpus}", end='\r')
except KeyboardInterrupt:
    print("\n\n🛑 クラスタをシャットダウン中...")
    ray.shutdown()
    print("✅ 終了しました")