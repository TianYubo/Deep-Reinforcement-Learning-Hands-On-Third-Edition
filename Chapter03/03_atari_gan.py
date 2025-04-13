#!/usr/bin/env python
import cv2
import time
import random
import argparse
import typing as tt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import torchvision.utils as vutils

import gymnasium as gym
from gymnasium import spaces

import numpy as np

log = gym.logger
log.set_level(gym.logger.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000

class InputWrapper(gym.ObservationWrapper):
    """
    对环境观测图像进行预处理：
    1. 将原始图像缩放到统一的尺寸（IMAGE_SIZE × IMAGE_SIZE）
    2. 将通道维度从最后一维移动到最前（从 HWC -> CHW），以符合 PyTorch 输入格式
    """

    def __init__(self, *args):
        # 调用父类构造函数初始化
        super(InputWrapper, self).__init__(*args)

        # 获取原始环境的 observation space，类型应为 Box（即图像类型的空间）
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)

        # 重新定义 observation_space，确保处理后的图像符合新的形状和数据类型
        self.observation_space = spaces.Box(
            self.observation(old_space.low),     # 最小值处理后对应新图像形状
            self.observation(old_space.high),    # 最大值处理后对应新图像形状
            dtype=np.float32                     # 将图像数据类型设置为 float32（适用于深度学习）
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        """
        重写 observation 方法，对每帧环境图像进行预处理。
        参数：
            observation: 从环境中获得的原始图像（格式为 H×W×C）
        返回：
            预处理后的图像（格式为 C×H×W，数据类型为 float32）
        """
        # 将图像缩放到目标尺寸（IMAGE_SIZE × IMAGE_SIZE）
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))

        # 将通道维从最后一维（H×W×C）移动到最前（C×H×W），以适应 PyTorch 的卷积输入格式
        new_obs = np.moveaxis(new_obs, 2, 0)

        # 转换为 float32 类型并返回
        return new_obs.astype(np.float32)



class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(envs: tt.List[gym.Env],
                    batch_size: int = BATCH_SIZE) -> tt.Generator[torch.Tensor, None, None]:
    """
    从多个 Gym 环境中不断采样图像，构造并返回标准化后的 batch 数据（[-1, 1] 区间）用于 GAN 训练。
    
    参数：
        envs: 包含多个已初始化 Gym 环境（必须经过 InputWrapper 处理）
        batch_size: 每个 batch 包含的图像数量，默认为 BATCH_SIZE

    返回：
        一个生成器，每次返回一个形状为 (B, C, H, W) 的 torch.Tensor，数值已归一化至 [-1, 1]
    """
    # 初始化 batch，先从每个环境中 reset 一帧图像作为初始数据
    batch = [e.reset()[0] for e in envs]

    # 无限循环从环境中随机采样数据（使用 lambda 构建一个迭代器）
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        # 随机选一个环境
        e = next(env_gen)

        # 对该环境采取随机动作（无监督采样）
        action = e.action_space.sample()

        # 执行一步动作，获取观测图像
        obs, reward, is_done, is_trunc, _ = e.step(action)

        # 简单滤波：排除全黑图（均值很低）避免干扰生成器学习
        if np.mean(obs) > 0.01:
            batch.append(obs)

        # 如果累积到 batch_size 数量
        if len(batch) == batch_size:
            # 转换为 NumPy 数组，并归一化到 [-1, 1] 区间
            batch_np = np.array(batch, dtype=np.float32)
            normed = batch_np * 2.0 / 255.0 - 1.0  # 原始图像像素值范围为 [0, 255]

            # 转换为 PyTorch 张量并 yield 出去
            yield torch.tensor(normed)

            # 清空 batch 为下次采样做准备
            batch.clear()

        # 如果环境结束（is_done）或被截断（is_trunc），重新 reset 环境
        if is_done or is_trunc:
            e.reset()



if __name__ == "__main__":
    # 解析命令行参数，--dev 指定使用的设备（如 cuda 或 cpu）
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.dev)

    # 初始化多个 Atari 游戏环境（经过 InputWrapper 封装）
    envs = [
        InputWrapper(gym.make(name))
        for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')
    ]
    shape = envs[0].observation_space.shape  # 获取图像的形状 (C, H, W)

    # 初始化判别器和生成器网络
    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)

    # 使用二分类交叉熵损失（生成器目标是骗过判别器，判别器目标是区分真假）
    objective = nn.BCELoss()

    # 使用 Adam 优化器训练生成器和判别器（betas 是 GAN 中常用的设置）
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # TensorBoard 可视化记录器
    writer = SummaryWriter()

    gen_losses = []  # 用于记录生成器的损失
    dis_losses = []  # 用于记录判别器的损失
    iter_no = 0      # 迭代次数记录

    # 生成标签向量（用于计算损失）：真实图像为 1，生成图像为 0
    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    ts_start = time.time()  # 记录训练起始时间

    # 主训练循环，从环境中不断获取图像 batch 进行训练
    for batch_v in iterate_batches(envs):
        # ========== 1. 生成器生成假图像 ==========
        # 构造输入的 latent 向量，大小为 (BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)  # 从标准正态分布中采样
        gen_input_v = gen_input_v.to(device)

        # 将真实图像移动到计算设备上
        batch_v = batch_v.to(device)

        # 生成假图像
        gen_output_v = net_gener(gen_input_v)

        # ========== 2. 判别器训练 ==========
        dis_optimizer.zero_grad()
        # 判别器输出真实图像的判断结果
        dis_output_true_v = net_discr(batch_v)
        # 判别器输出生成图像（detach）的判断结果
        dis_output_fake_v = net_discr(gen_output_v.detach())
        # 判别器损失 = 真实图像判为真 + 假图像判为假
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                   objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # ========== 3. 生成器训练 ==========
        gen_optimizer.zero_grad()
        # 重新判断生成图像（注意不 detach，因为需要更新生成器）
        dis_output_v = net_discr(gen_output_v)
        # 生成器的目标是让判别器认为它生成的图像是真实的
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        # ========== 4. 记录日志与保存图像 ==========
        iter_no += 1

        # 每隔 REPORT_EVERY_ITER 次输出当前损失
        if iter_no % REPORT_EVERY_ITER == 0:
            dt = time.time() - ts_start
            log.info("Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
            ts_start = time.time()
            # 记录到 TensorBoard
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []

        # 每隔 SAVE_IMAGE_EVERY_ITER 次保存图像到 TensorBoard
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            # 保存生成图像
            img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
            writer.add_image("fake", img, iter_no)
            # 保存真实图像
            img = vutils.make_grid(batch_v.data[:64], normalize=True)
            writer.add_image("real", img, iter_no)