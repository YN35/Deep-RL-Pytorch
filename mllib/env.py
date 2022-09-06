
from turtle import forward
import gym

def get_env_class(name):
    return getattr(getattr(__import__('mllib'), 'env'), name)

class CartPole_v1():
    def __init__(self, param) -> None:
        self.enable_render = param.enable_render
        self.env = gym.make('CartPole-v1')
        param.observation_space = self.env.observation_space
        param._action_space = self.env.action_space
        print('[ENV] observation space is ' + str(param.observation_space))
        print('[ENV] action space is ' + str(param._action_space))
        
    def reset(self):
        obs = self.env.reset()
        return obs
        
    def step(self, action):
        if self.enable_render:
            self.env.render()
        
        observation, reward, done, info = self.env.step(action)
        
        return observation, reward, done, info