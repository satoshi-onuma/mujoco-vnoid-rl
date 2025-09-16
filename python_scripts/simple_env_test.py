import sys
import os

def test_module_import():
    """Step 1: C++モジュールのインポートテスト"""
    print("=== Step 1: モジュールインポートテスト ===")
    
    build_path = os.path.abspath("./build/controller/vnoid_rl_env")
    sys.path.append(build_path)
    
    try:
        import vnoid_rl_env
        print("✅ C++モジュールのインポート成功")
        return True
    except ImportError as e:
        print(f"❌ C++モジュールのインポート失敗: {e}")
        return False

def test_single_env_creation():
    """Step 2: 単一環境の作成テスト"""
    print("\n=== Step 2: 単一環境作成テスト ===")
    
    try:
        import vnoid_rl_env
        model_path = "model/sample_robot/sample_robot_mujoco.xml"
        
        print("C++環境を作成中...")
        env = vnoid_rl_env.VnoidEnv(model_path)
        print("✅ 単一環境の作成成功")
        
        # 環境のクリーンアップ
        del env
        print("✅ 環境のクリーンアップ成功")
        return True
        
    except Exception as e:
        print(f"❌ 単一環境の作成失敗: {e}")
        return False

def test_env_basic_operations():
    """Step 3: 基本操作テスト"""
    print("\n=== Step 3: 基本操作テスト ===")
    
    try:
        import vnoid_rl_env
        model_path = "model/sample_robot/sample_robot_mujoco.xml"
        
        env = vnoid_rl_env.VnoidEnv(model_path)
        
        # リセットテスト
        print("環境をリセット中...")
        obs = env.reset()
        print(f"✅ リセット成功。観測次元: {obs.shape}")
        
        # ステップテスト
        print("1ステップ実行中...")
        action = [0.01, -0.01]
        obs, reward, terminated, info = env.step(action)
        print(f"✅ ステップ成功。報酬: {reward}, 終了: {terminated}")
        
        del env
        return True
        
    except Exception as e:
        print(f"❌ 基本操作テスト失敗: {e}")
        return False

def test_rendering():
    """Step 4: レンダリングテスト"""
    print("\n=== Step 4: レンダリングテスト ===")
    
    try:
        import vnoid_rl_env
        model_path = "model/sample_robot/sample_robot_mujoco.xml"
        
        env = vnoid_rl_env.VnoidEnv(model_path)
        env.reset()
        
        print("レンダリング実行中...")
        frame = env.render()
        
        if frame is not None:
            print(f"✅ レンダリング成功。フレーム形状: {frame.shape}")
            return True
        else:
            print("❌ レンダリング失敗: フレームがNone")
            return False
            
        del env
        
    except Exception as e:
        print(f"❌ レンダリングテスト失敗: {e}")
        return False

def test_python_wrapper():
    """Step 5: Python環境ラッパーテスト"""
    print("\n=== Step 5: Python環境ラッパーテスト ===")
    
    try:
        from my_humanoid_env import HumanoidVnoidEnv
        
        print("Python環境ラッパーを作成中...")
        env = HumanoidVnoidEnv()
        
        obs, info = env.reset()
        print(f"✅ Python環境リセット成功。観測次元: {obs.shape}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Python環境ステップ成功。報酬: {reward}")
        
        frame = env.render()
        if frame is not None:
            print(f"✅ Python環境レンダリング成功。フレーム形状: {frame.shape}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Python環境ラッパーテスト失敗: {e}")
        return False

def main():
    """メインテスト関数"""
    print("録画機能の段階的テストを開始します...\n")
    
    tests = [
        test_module_import,
        test_single_env_creation,
        test_env_basic_operations,
        test_rendering,
        test_python_wrapper
    ]
    
    for i, test in enumerate(tests, 1):
        success = test()
        if not success:
            print(f"\n❌ テスト{i}で失敗。後続のテストをスキップします。")
            return False
    
    print("\n✅ すべてのテストが成功しました！")
    print("録画スクリプトを実行する準備ができています。")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)