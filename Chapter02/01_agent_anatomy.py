import random
from typing import List


class Environment:
    """环境类：表示强化学习中的环境"""
    def __init__(self):
        # 初始化环境，设置剩余步数为10
        self.steps_left = 10

    def get_observation(self) -> List[float]:
        # 获取当前环境的观察值（状态）
        # 这里简单返回一个固定的观察值[0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        # 获取可用的动作列表
        # 这里只有两个可能的动作：0和1
        return [0, 1]

    def is_done(self) -> bool:
        # 判断环境是否结束
        # 当剩余步数为0时，环境结束
        return self.steps_left == 0

    def action(self, action: int) -> float:
        # 执行动作并返回奖励
        if self.is_done():
            # 如果环境已经结束，抛出异常
            raise Exception("Game is over")
        # 每执行一个动作，剩余步数减1
        self.steps_left -= 1
        # 返回一个0到1之间的随机奖励值
        return random.random()


class Agent:
    """智能体类：表示在环境中学习和行动的主体"""
    def __init__(self):
        # 初始化智能体，总奖励设为0
        self.total_reward = 0.0

    def step(self, env: Environment):
        # 智能体在环境中执行一步
        # 获取当前环境的观察值
        current_obs = env.get_observation()
        
        # 获取可用的动作列表
        actions = env.get_actions()
        
        # 随机选择一个动作并执行，获取奖励
        # 这里使用了随机策略(random policy)
        reward = env.action(random.choice(actions))
        
        # 累加奖励
        self.total_reward += reward


if __name__ == "__main__":
    # 创建环境实例
    env = Environment()
    # 创建智能体实例
    agent = Agent()

    # 主循环：只要环境没有结束，就继续执行
    while not env.is_done():
        # 智能体在环境中执行一步
        agent.step(env)

    # 输出智能体获得的总奖励
    print("Total reward got: %.4f" % agent.total_reward)
