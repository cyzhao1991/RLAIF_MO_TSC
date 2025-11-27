import os
# os.environ["OPENAI_API_KEY"]="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"
# os.environ["GEMINI_API_KEY"]="AIzaSyAR5hGVYhG8FAItSPyIoDWLc12YJ4DbaRc"
# os.environ['SUMO_HOME'] = '/home/jovyan/sumo/sumo-install'
# os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/home/jovyan/sumo/icu-install/lib:/home/jovyan/sumo/xerces-install/lib'
# os.environ['PATH'] = os.environ.get('PATH', '') + ':/home/jovyan/sumo/sumo-install/bin'
from reward_model import RewardModel

from omegaconf import OmegaConf
import hydra
from utils import make_classic_control_env, make_metaworld_env
import metaworld.envs.mujoco.env_dict as _env_dict

import sumo_rl
import traci
import gymnasium
import numpy as np
import pandas as pd
import pickle
import wandb
import datetime, time
from tqdm import tqdm

from gym.wrappers.time_limit import TimeLimit
from rlkit.envs.wrappers import NormalizedBoxEnv




# wandb.init(
#     project = "rlaif_traffic",
#     name = 'reward_learning',
# )


ts = time.time()
time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
# run = wandb.init(
#     project = "rlaif_traffic",
#     group = "reward_learning",
#     name = 'proxy'+time_string,
# )

# log_dir = os.path.join('/home/jovyan/projects/sumo_agent/reward_learning_logs', run.id)
# model_dir = os.path.join(log_dir, 'models')
# os.makedirs(log_dir, exist_ok=True)
# os.makedirs(model_dir, exist_ok=True)

reward_model = RewardModel(29, 4, 
    ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
    max_size=100, activation='tanh', capacity=5e5,  
    large_batch=1, label_margin=0.0, 
    teacher_beta=-1, teacher_gamma=1, 
    teacher_eps_mistake=0, 
    teacher_eps_skip=0, 
    teacher_eps_equal=0,

    # vlm related params
    vlm_label=0,
    env_name="tsc_single-intersection",
    vlm="text_only",
    clip_prompt=None,
    log_dir=None,
    # log_dir=log_dir,
    flip_vlm_label=False,
    save_query_interval=1,
    cached_label_path=None,
    llm_label=0,

    # image based reward
    reward_model_layers=3,
    reward_model_H=256,
    image_reward=False,
    image_height=128,
    image_width=128,
    resize_factor=1,
    resnet=False,
    conv_kernel_sizes=[5, 3, 3 ,3],
    conv_n_channels=[16, 32, 64, 128],
    conv_strides=[3, 2, 2, 2],
)



# ppo_offline_data_path = '/home/jovyan/projects/sumo_agent/outputs/2way-single-intersection/ppo/buffers'
ppo_offline_data_path = '/home/czhao/workspace/sumo_rl/pressure' #jnx7nhbh/buffers'
# /home/jovyan/projects/sumo_agent/outputs/single-intersection/ppo/rollout_buffer.pkl
with open(os.path.join(ppo_offline_data_path, 'rollout_buffer.pkl'), 'rb') as f:
  data = pickle.load(f)

total_data_count = data['obs'].shape[0]

if total_data_count > reward_model.capacity:
  idxs = np.random.choice(np.arange(total_data_count), size=reward_model.capacity, replace=False)
else:
  idxs = np.arange(total_data_count)


# total_data_count = min(reward_model.capacity, data['obs'].shape[0])
for i in tqdm(idxs):
  o, a, r, d, inf = data['obs'][i], data['act'][i], data['rew'][i], data['done'][i], data['info'][i]
# for o, a, r, d, i in tqdm(list(zip(data['obs'], data['act'], data['rew'], data['done'], data['info']))):
  reward_model.add_data(o, a, r, d, inf)

with open(os.path.join(ppo_offline_data_path, 'reward_model.pkl'), 'wb') as f:
    pickle.dump(reward_model, f)


# for i in range(500):
#   _ = reward_model.uniform_sampling()
#   for epoch in range(5):
#     train_acc, train_rational_acc = reward_model.train_reward()
#     total_acc = np.mean(train_acc)
#     total_rational_acc = np.mean(train_rational_acc)
#     print("Reward function is updated!! ")
#     print("ACC: " + str(total_acc))
#     print("Rational Acc: " + str(total_rational_acc))
#   wandb.log({
#     "train/reward_learning_acc": total_acc,
#     "train/reward_learning_rational_acc": total_rational_acc,
#   }, i)

#   if i % 10 == 0:
#     reward_model.save(model_dir, i)