import torch


class Epx:
    def __init__(self, device, profile_name, exp_buffer_size) -> None:
        """Experience buffer. You can set buffer size."""
        
        self.device = device
        self.exp_buffer_size = exp_buffer_size
        self.reward_sum = 0
        
        self.rewards = torch.empty(self.exp_buffer_size).to(self.device)
        self.done = torch.empty(self.exp_buffer_size, dtype=torch.bool).to(self.device)
        
        getattr(self,profile_name)()
        
    def roll(self):
        """Roll the buffer. The oldest experience will be move to first index."""
        for tensors in self.__dict__.items():
            if type(tensors[1]) == torch.Tensor:
                self.__dict__[tensors[0]] = tensors[1].roll(1, dims=0)
        
        
    def DQN(self):
        self.esti_qmax = torch.empty(self.exp_buffer_size).to(self.device)