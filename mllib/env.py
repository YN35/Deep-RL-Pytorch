
from turtle import forward
import gym

def get_env_class(name):
    return getattr(getattr(__import__('mllib'), 'env'), name)

class CartPole_v0():
    def __init__(self, enable_render=False) -> None:
        self.enable_render = enable_render
        self.env = gym.make('CartPole-v0')
        print('[ENV] action space is ' + str(self.env.action_space.sample()))
        
        
    def forward(self, action):
        if self.enable_render:
            self.env.render()
        
        observation, reward, done, info = self.env.step(action)
        
        return observation, reward, done, info