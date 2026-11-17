"""D4a 前馈容量匹配控制器：唯一不具有跨 tick 内部状态的独立 BC 基线。"""
from torch import nn

OBS_DIM = 259
ACTION_DIM = 2
WIDTH1 = 256
WIDTH2 = 185


class FeedforwardController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, WIDTH1), nn.Tanh(),
            nn.Linear(WIDTH1, WIDTH2), nn.Tanh(),
            nn.Linear(WIDTH2, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, observation):
        # 支持 (batch, steps, 259)；每个时刻独立处理，没有跨步隐状态。
        assert observation.shape[-1] == OBS_DIM
        return self.net(observation)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
