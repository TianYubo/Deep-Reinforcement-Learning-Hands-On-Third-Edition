#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import typing as tt
import random
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 定义常量
HIDDEN_SIZE = 128  # 神经网络隐藏层大小
BATCH_SIZE = 100  # 每批次的训练数据量
PERCENTILE = 30  # 用于筛选精英样本的百分位数
GAMMA = 0.9  # 折扣因子，用于计算未来奖励的现值

# 定义一个包装器，将离散的观察空间转换为 one-hot 编码
class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space,
                          gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int,
                 n_actions: int):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(hidden_size, n_actions)  # 隐藏层到输出层
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# 定义数据类，用于存储每一步的观察和动作
@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

# 定义数据类，用于存储整个回合的奖励和步骤
@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]

# 生成批次数据，用于训练
def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> \
        tt.Generator[tt.List[Episode], None, None]:
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.tensor(obs, dtype=torch.float32)
        act_probs_v = sm(net(obs_v.unsqueeze(0)))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs

# 筛选精英样本，用于训练
def filter_batch(batch: tt.List[Episode], percentile: float) -> \
        tt.Tuple[tt.List[Episode], tt.List[np.ndarray], tt.List[int], float]:
    reward_fun = lambda s: s.reward * (GAMMA ** len(s.steps))
    disc_rewards = list(map(reward_fun, batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    elite_batch: tt.List[Episode] = []

    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound

# 主函数
if __name__ == "__main__":
    random.seed(12345)  # 设置随机种子，确保结果可复现
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1"))  # 创建环境并应用包装器
    obs_size = env.observation_space.shape[0]  # 获取观察空间的大小
    n_actions = env.action_space.n  # 获取动作空间的大小

    net = Net(obs_size, HIDDEN_SIZE, n_actions)  # 初始化神经网络
    objective = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)  # 定义优化器
    writer = SummaryWriter(comment="-frozenlake-tweaked")  # 初始化 TensorBoard 记录器

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)  # 这里 full_batch 就是保留的精英样本
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(np.vstack(obs))
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, rw_mean=%.3f, "
              "rw_bound=%.3f, batch=%d" % (
            iter_no, loss_v.item(), reward_mean,
            reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        if reward_mean > 0.8:
            print("Solved!")
            break
    writer.close()
