import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 基于Gym环境的DCGAN训练\n",
                "本Notebook实现了一个使用Gym环境图像作为训练数据的DCGAN（Deep Convolutional GAN）。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 安装依赖库（如果未安装）\n",
                "!pip install gymnasium[atari] torch torchvision tensorboard opencv-python"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import cv2\n",
                "import time\n",
                "import random\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.optim as optim\n",
                "from torch.utils.tensorboard.writer import SummaryWriter\n",
                "import torchvision.utils as vutils\n",
                "import gymnasium as gym\n",
                "from gymnasium import spaces\n",
                "import numpy as np"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 参数设置"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "LATENT_VECTOR_SIZE = 100\n",
                "DISCR_FILTERS = 64\n",
                "GENER_FILTERS = 64\n",
                "BATCH_SIZE = 16\n",
                "IMAGE_SIZE = 64\n",
                "LEARNING_RATE = 0.0001\n",
                "REPORT_EVERY_ITER = 100\n",
                "SAVE_IMAGE_EVERY_ITER = 1000\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 输入封装器\n",
                "将环境图像缩放为指定尺寸，并将通道维移动到前面以符合PyTorch格式。"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class InputWrapper(gym.ObservationWrapper):\n",
                "    def __init__(self, *args):\n",
                "        super(InputWrapper, self).__init__(*args)\n",
                "        old_space = self.observation_space\n",
                "        assert isinstance(old_space, spaces.Box)\n",
                "        self.observation_space = spaces.Box(\n",
                "            self.observation(old_space.low), self.observation(old_space.high),\n",
                "            dtype=np.float32\n",
                "        )\n",
                "\n",
                "    def observation(self, observation):\n",
                "        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))\n",
                "        new_obs = np.moveaxis(new_obs, 2, 0)\n",
                "        return new_obs.astype(np.float32)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 判别器模型（Discriminator）"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class Discriminator(nn.Module):\n",
                "    def __init__(self, input_shape):\n",
                "        super(Discriminator, self).__init__()\n",
                "        self.conv_pipe = nn.Sequential(\n",
                "            nn.Conv2d(input_shape[0], DISCR_FILTERS, 4, 2, 1),\n",
                "            nn.ReLU(),\n",
                "            nn.Conv2d(DISCR_FILTERS, DISCR_FILTERS*2, 4, 2, 1),\n",
                "            nn.BatchNorm2d(DISCR_FILTERS*2),\n",
                "            nn.ReLU(),\n",
                "            nn.Conv2d(DISCR_FILTERS*2, DISCR_FILTERS*4, 4, 2, 1),\n",
                "            nn.BatchNorm2d(DISCR_FILTERS*4),\n",
                "            nn.ReLU(),\n",
                "            nn.Conv2d(DISCR_FILTERS*4, DISCR_FILTERS*8, 4, 2, 1),\n",
                "            nn.BatchNorm2d(DISCR_FILTERS*8),\n",
                "            nn.ReLU(),\n",
                "            nn.Conv2d(DISCR_FILTERS*8, 1, 4, 1, 0),\n",
                "            nn.Sigmoid()\n",
                "        )\n",
                "\n",
                "    def forward(self, x):\n",
                "        return self.conv_pipe(x).view(-1, 1).squeeze(dim=1)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 生成器模型（Generator）"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class Generator(nn.Module):\n",
                "    def __init__(self, output_shape):\n",
                "        super(Generator, self).__init__()\n",
                "        self.pipe = nn.Sequential(\n",
                "            nn.ConvTranspose2d(LATENT_VECTOR_SIZE, GENER_FILTERS*8, 4, 1, 0),\n",
                "            nn.BatchNorm2d(GENER_FILTERS*8),\n",
                "            nn.ReLU(),\n",
                "            nn.ConvTranspose2d(GENER_FILTERS*8, GENER_FILTERS*4, 4, 2, 1),\n",
                "            nn.BatchNorm2d(GENER_FILTERS*4),\n",
                "            nn.ReLU(),\n",
                "            nn.ConvTranspose2d(GENER_FILTERS*4, GENER_FILTERS*2, 4, 2, 1),\n",
                "            nn.BatchNorm2d(GENER_FILTERS*2),\n",
                "            nn.ReLU(),\n",
                "            nn.ConvTranspose2d(GENER_FILTERS*2, GENER_FILTERS, 4, 2, 1),\n",
                "            nn.BatchNorm2d(GENER_FILTERS),\n",
                "            nn.ReLU(),\n",
                "            nn.ConvTranspose2d(GENER_FILTERS, output_shape[0], 4, 2, 1),\n",
                "            nn.Tanh()\n",
                "        )\n",
                "\n",
                "    def forward(self, x):\n",
                "        return self.pipe(x)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("/mnt/share_disk/tianyubo/Deep-Reinforcement-Learning-Hands-On-Third-Edition/Chapter03/dcgan_gym_notebook.ipynb", "w") as f:
    json.dump(notebook, f)
    

# 继续添加 Notebook 的后续部分，包括 batch 迭代器和训练主循环
additional_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 数据迭代器函数 iterate_batches\n",
            "用于从多个Gym环境中采样图像并构建batch。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def iterate_batches(envs, batch_size=BATCH_SIZE):\n",
            "    batch = [e.reset()[0] for e in envs]\n",
            "    env_gen = iter(lambda: random.choice(envs), None)\n",
            "\n",
            "    while True:\n",
            "        e = next(env_gen)\n",
            "        action = e.action_space.sample()\n",
            "        obs, reward, is_done, is_trunc, _ = e.step(action)\n",
            "        if np.mean(obs) > 0.01:\n",
            "            batch.append(obs)\n",
            "        if len(batch) == batch_size:\n",
            "            batch_np = np.array(batch, dtype=np.float32)\n",
            "            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)\n",
            "            batch.clear()\n",
            "        if is_done or is_trunc:\n",
            "            e.reset()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 模型初始化与训练循环\n",
            "包括判别器和生成器的训练、损失计算和TensorBoard记录。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 初始化环境\n",
            "envs = [\n",
            "    InputWrapper(gym.make(name))\n",
            "    for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')\n",
            "]\n",
            "shape = envs[0].observation_space.shape\n",
            "\n",
            "net_discr = Discriminator(input_shape=shape).to(device)\n",
            "net_gener = Generator(output_shape=shape).to(device)\n",
            "\n",
            "objective = nn.BCELoss()\n",
            "gen_optimizer = optim.Adam(net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))\n",
            "dis_optimizer = optim.Adam(net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))\n",
            "writer = SummaryWriter()\n",
            "\n",
            "gen_losses = []\n",
            "dis_losses = []\n",
            "iter_no = 0\n",
            "true_labels_v = torch.ones(BATCH_SIZE, device=device)\n",
            "fake_labels_v = torch.zeros(BATCH_SIZE, device=device)\n",
            "ts_start = time.time()\n",
            "\n",
            "# 训练主循环\n",
            "for batch_v in iterate_batches(envs):\n",
            "    gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)\n",
            "    batch_v = batch_v.to(device)\n",
            "    gen_output_v = net_gener(gen_input_v)\n",
            "\n",
            "    # 训练判别器\n",
            "    dis_optimizer.zero_grad()\n",
            "    dis_output_true_v = net_discr(batch_v)\n",
            "    dis_output_fake_v = net_discr(gen_output_v.detach())\n",
            "    dis_loss = objective(dis_output_true_v, true_labels_v) + \\\n",
            "               objective(dis_output_fake_v, fake_labels_v)\n",
            "    dis_loss.backward()\n",
            "    dis_optimizer.step()\n",
            "    dis_losses.append(dis_loss.item())\n",
            "\n",
            "    # 训练生成器\n",
            "    gen_optimizer.zero_grad()\n",
            "    dis_output_v = net_discr(gen_output_v)\n",
            "    gen_loss_v = objective(dis_output_v, true_labels_v)\n",
            "    gen_loss_v.backward()\n",
            "    gen_optimizer.step()\n",
            "    gen_losses.append(gen_loss_v.item())\n",
            "\n",
            "    iter_no += 1\n",
            "    if iter_no % REPORT_EVERY_ITER == 0:\n",
            "        dt = time.time() - ts_start\n",
            "        print(f\"Iter {iter_no} in {dt:.2f}s: gen_loss={np.mean(gen_losses):.3e}, dis_loss={np.mean(dis_losses):.3e}\")\n",
            "        ts_start = time.time()\n",
            "        writer.add_scalar(\"gen_loss\", np.mean(gen_losses), iter_no)\n",
            "        writer.add_scalar(\"dis_loss\", np.mean(dis_losses), iter_no)\n",
            "        gen_losses = []\n",
            "        dis_losses = []\n",
            "\n",
            "    if iter_no % SAVE_IMAGE_EVERY_ITER == 0:\n",
            "        img = vutils.make_grid(gen_output_v.data[:64], normalize=True)\n",
            "        writer.add_image(\"fake\", img, iter_no)\n",
            "        img = vutils.make_grid(batch_v.data[:64], normalize=True)\n",
            "        writer.add_image(\"real\", img, iter_no)"
        ]
    }
]

# 读取原有 notebook，追加 cell
with open("/mnt/share_disk/tianyubo/Deep-Reinforcement-Learning-Hands-On-Third-Edition/Chapter03/dcgan_gym_notebook.ipynb", "r") as f:
    notebook = json.load(f)

notebook["cells"].extend(additional_cells)

# 写回 notebook 文件
with open("/mnt/share_disk/tianyubo/Deep-Reinforcement-Learning-Hands-On-Third-Edition/Chapter03/dcgan_gym_notebook.ipynb", "w") as f:
    json.dump(notebook, f)
