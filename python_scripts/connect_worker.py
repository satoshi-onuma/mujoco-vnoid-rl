# connect_worker.py
# Worker NodeとしてHead Nodeに接続するスクリプト

import ray
import time
import os
import sys

print("=" * 70)
print("🔌 Ray Worker - Head Nodeに接続")
print("=" * 70)

# 環境変数からHead Nodeのアドレスを取得
ray_address = os.environ.get('RAY_ADDRESS')

if not ray_address:
    print("\n❌ エラー: RAY_ADDRESS環境変数が設定されていません")
    print("\n以下を実行してから再度試してください:")
    print("  export RAY_ADDRESS='<Head NodeのIP>:6379'")
    print("\n例:")
    print("  export RAY_ADDRESS='10.32.41.106:6379'")
    print("  python connect_worker.py")
    sys.exit(1)

print(f"📍 接続先: {ray_address}")
print(f"🔄 接続中...\n")

try:
    # Head Nodeに接続
    ray.init(address=ray_address)
    
    print("✅ 接続成功！\n")
    
    # このWorkerの情報を表示
    print("このWorkerの情報:")
    print(f"  ノードIP: {ray.get_runtime_context().node_ip}")
    resources = ray.available_resources()
    print(f"  CPU: {resources.get('CPU', 0)}")
    print(f"  メモリ: {resources.get('memory', 0) / (1024**3):.1f} GB")
    
    # クラスタ全体の情報
    print(f"\nクラスタ全体:")
    print(f"  総ノード数: {len(ray.nodes())}")
    print(f"  総CPU: {ray.cluster_resources().get('CPU', 0)}")
    
    print("\n" + "=" * 70)
    print("✅ Head Nodeに接続しました")
    print("=" * 70)
    print("このターミナルは開いたままにしてください")
    print("Head Nodeで学習スクリプトを実行すると、")
    print("このWorkerが自動的にタスクを実行します")
    print("\nCtrl+C で切断します")
    print("=" * 70 + "\n")
    
    # 接続を維持
    last_nodes = len(ray.nodes())
    while True:
        time.sleep(5)
        current_nodes = len(ray.nodes())
        if current_nodes != last_nodes:
            print(f"[ノード数変更] {last_nodes} → {current_nodes}")
            last_nodes = current_nodes
        print(f"[{time.strftime('%H:%M:%S')}] 接続中... ノード数: {current_nodes}", end='\r')

except KeyboardInterrupt:
    print("\n\n🛑 切断中...")
    ray.shutdown()
    print("✅ 切断完了")
    
except Exception as e:
    print(f"\n❌ エラーが発生しました: {e}")
    print("\n考えられる原因:")
    print("  1. Head Nodeが起動していない")
    print("  2. ネットワーク接続の問題（ファイアウォール等）")
    print("  3. RAY_ADDRESSが間違っている")
    print("\nHead Nodeで以下を確認してください:")
    print("  ray status")
    import traceback
    traceback.print_exc()
    sys.exit(1)