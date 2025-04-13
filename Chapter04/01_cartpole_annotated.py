#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
from dataclasses import dataclass
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128  # 神经网络隐藏层大小
BATCH_SIZE = 16    # 每批次训练的样本数量
PERCENTILE = 70    # 用于筛选优质样本的百分位阈值


class Net(nn.Module):
    """定义策略网络，将观察空间映射到动作概率"""
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int, num_layers: int = 2):
        """
        初始化网络
        
        参数:
            obs_size: int - 观察空间的维度
            hidden_size: int - 隐藏层的大小
            n_actions: int - 动作空间的维度
            num_layers: int - 总的线性层数量，默认为2（输入层和输出层）。隐藏层数量 = num_layers - 2
        """
        super(Net, self).__init__()
        
        # 检查层数是否合法
        if num_layers < 2:
            raise ValueError("总层数必须至少为2（输入层和输出层）")
        
        # 构建网络层
        layers = []
        
        # 添加输入层
        layers.append(nn.Linear(obs_size, hidden_size))
        layers.append(nn.ReLU())
        
        # 添加隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # 添加输出层
        layers.append(nn.Linear(hidden_size, n_actions))
        
        # 创建Sequential模型
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """前向传播"""
        return self.net(x)


@dataclass
class EpisodeStep:
    """记录单步交互的数据结构"""
    observation: np.ndarray  # 观察状态
    action: int              # 执行的动作


@dataclass
class Episode:
    """记录整个回合的数据结构"""
    reward: float                # 回合总奖励
    steps: tt.List[EpisodeStep]  # 回合中的所有步骤


def iterate_batches(env: gym.Env, net: Net, batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:
    """
    生成训练批次数据的生成器
    
    Args:
        env: 强化学习环境
        net: 策略网络
        batch_size: 批次大小
        
    Yields:
        包含多个回合数据的批次
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()  # 重置环境，获取初始观察
    sm = nn.Softmax(dim=1)  # Softmax函数，用于将网络输出转换为概率分布
    
    while True:
        # 将观察转换为张量
        obs_v = torch.tensor(obs, dtype=torch.float32)
        # 获取动作概率
        act_probs_v = sm(net(obs_v.unsqueeze(0)))
        act_probs = act_probs_v.data.numpy()[0]
        # 根据概率分布随机选择动作
        action = np.random.choice(len(act_probs), p=act_probs)
        # 执行动作
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        # 累加奖励
        episode_reward += float(reward)
        # 记录这一步
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        
        # 如果回合结束
        if is_done or is_trunc:
            # 创建回合记录并添加到批次
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            # 重置回合数据
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            # 如果批次已满，返回批次数据
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        obs = next_obs  # 更新观察状态


def filter_batch(batch: tt.List[Episode], percentile: float) -> \
        tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    """
    根据奖励筛选优质样本用于训练
    
    Args:
        batch: 回合批次
        percentile: 筛选的百分位阈值
        
    Returns:
        训练观察数据、动作数据、奖励阈值、平均奖励
    """
    # 提取所有回合的奖励
    # rewards = list(map(lambda s: s.reward, batch))
    rewards = [s.reward for s in batch]
    # 计算奖励阈值（只有高于此阈值的回合才会被用于训练）
    reward_bound = float(np.percentile(rewards, percentile))
    # 计算平均奖励
    reward_mean = float(np.mean(rewards))

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    # 筛选高奖励回合的数据
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        # 提取观察和动作
        # train_obs.extend(map(lambda step: step.observation, episode.steps))
        # train_act.extend(map(lambda step: step.action, episode.steps))
        
        train_obs.extend([step.observation for step in episode.steps])
        train_act.extend([step.action for step in episode.steps])

    # 转换为张量
    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    # 创建CartPole环境
    # env = gym.make("CartPole-v1")
    
    env = gym.make("CartPole-v1", render_mode="rgb_array") 
    env = gym.wrappers.RecordVideo(env, video_folder="video")
    
    assert env.observation_space.shape is not None
    # 获取观察空间大小
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    # 获取动作空间大小
    n_actions = int(env.action_space.n)

    # 创建策略网络
    net = Net(obs_size, HIDDEN_SIZE, n_actions, num_layers=4)
    print(net)
    # 定义损失函数（交叉熵）
    objective = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    # 创建TensorBoard记录器
    writer = SummaryWriter(comment="-cartpole")

    # 训练循环
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # 筛选优质样本
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        action_scores_v = net(obs_v)
        # 计算损失
        loss_v = objective(action_scores_v, acts_v)
        # 反向传播
        loss_v.backward()
        # 更新参数
        optimizer.step()
        
        # 打印训练信息
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        # 记录指标到TensorBoard
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        
        # 如果平均奖励超过475，认为问题已解决
        if reward_m > 475:
            print("Solved!")
            break
            
    writer.close()
