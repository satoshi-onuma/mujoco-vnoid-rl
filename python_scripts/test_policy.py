import torch
import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
import json
import time

from ray import tune
import torch
from ray.rllib.core.rl_module import RLModule
from pathlib import Path
from pprint import pprint


dummy_obs = torch.zeros(1, 16)

checkpoint_dir = os.path.abspath("./humanoid_vnoid_checkpoint_id3")
if not os.path.exists(checkpoint_dir):
    print(f"\n❌ エラー: チェックポイントが見つかりません")
    print(f"パス: {checkpoint_dir}")
    print("\n先に学習を実行してください:")
    print("  python train_humanoid.py")
    exit(1)

# ★ os.path.join()でパスを結合
rl_module_path = os.path.join(
    checkpoint_dir,
    "learner_group",
    "learner",
    "rl_module",
    "default_policy"
)

try:
    print("\n📥 RLModuleをロード中...")
    rl_module = RLModule.from_checkpoint(rl_module_path)
    
    # #region agent log
    with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
        params_grad_status = {name: param.requires_grad for name, param in rl_module.named_parameters()}
        buffers_grad_status = {name: buf.requires_grad if isinstance(buf, torch.Tensor) else False for name, buf in rl_module.named_buffers()}
        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_1","timestamp":int(time.time()*1000),"location":"test_policy.py:39","message":"RLModule loaded - param/buffer grad status","data":{"training_mode":rl_module.training,"num_params":len(list(rl_module.parameters())),"num_buffers":len(list(rl_module.buffers())),"sample_params_grad":{k:v for k,v in list(params_grad_status.items())[:5]},"sample_buffers_grad":{k:v for k,v in list(buffers_grad_status.items())[:5]}},"runId":"post-fix","hypothesisId":"A_C_E"})+'\n')
    # #endregion
    
    # モジュールを評価モードに設定し、勾配追跡を無効化
    rl_module.eval()
    for param in rl_module.parameters():
        param.requires_grad = False
    
    # #region agent log
    with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
        params_grad_status_after = {name: param.requires_grad for name, param in rl_module.named_parameters()}
        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_1b","timestamp":int(time.time()*1000),"location":"test_policy.py:54","message":"After eval() and requires_grad=False","data":{"training_mode":rl_module.training,"sample_params_grad":{k:v for k,v in list(params_grad_status_after.items())[:5]}},"runId":"post-fix","hypothesisId":"FIX_A_E"})+'\n')
    # #endregion
    
except Exception as e:
    print(f"❌ ポリシーのロード失敗: {e}")
    exit(1)

try:
    # #region agent log
    with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_2","timestamp":int(time.time()*1000),"location":"test_policy.py:52","message":"Before forward_inference test","data":{"dummy_obs_shape":list(dummy_obs.shape),"dummy_obs_requires_grad":dummy_obs.requires_grad},"runId":"post-fix","hypothesisId":"B"})+'\n')
    # #endregion
    
    with torch.no_grad():
        test_output = rl_module.forward_inference({"obs": dummy_obs})
        
        # #region agent log
        with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
            action_dist_inputs = test_output["action_dist_inputs"]
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_3","timestamp":int(time.time()*1000),"location":"test_policy.py:62","message":"After forward_inference test","data":{"output_keys":list(test_output.keys()),"action_dist_inputs_shape":list(action_dist_inputs.shape),"action_dist_inputs_requires_grad":action_dist_inputs.requires_grad,"action_dist_inputs_is_leaf":action_dist_inputs.is_leaf},"runId":"post-fix","hypothesisId":"B"})+'\n')
        # #endregion
        
        # #region agent log
        with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_4","timestamp":int(time.time()*1000),"location":"test_policy.py:70","message":"Before torch.jit.trace","data":{"module_training_mode":rl_module.training},"runId":"post-fix","hypothesisId":"D_E"})+'\n')
        # #endregion
        
        traced = torch.jit.trace(
            lambda obs: rl_module.forward_inference({"obs": obs})["action_dist_inputs"][0, :2],
            dummy_obs
        )
        
        # #region agent log
        with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_6","timestamp":int(time.time()*1000),"location":"test_policy.py:81","message":"torch.jit.trace succeeded","data":{"traced_type":str(type(traced))},"runId":"post-fix","hypothesisId":"SUCCESS"})+'\n')
        # #endregion
        
    traced.save("policy.pt")
    print("成功")
    
    # #region agent log
    with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_7","timestamp":int(time.time()*1000),"location":"test_policy.py:90","message":"Policy saved successfully","data":{"file":"policy.pt"},"runId":"post-fix","hypothesisId":"SUCCESS"})+'\n')
    # #endregion
    
except Exception as e:
    # #region agent log
    with open('/home/satoshi/vnoid-mujoco/.cursor/debug.log', 'a') as f:
        f.write(json.dumps({"id":f"log_{int(time.time()*1000)}_5","timestamp":int(time.time()*1000),"location":"test_policy.py:84","message":"Exception caught","data":{"error_type":type(e).__name__,"error_message":str(e)},"runId":"post-fix","hypothesisId":"A_B_C_D_E"})+'\n')
    # #endregion
    print(f"失敗: {e}")