from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
import math

torch, nn = try_import_torch()


class FixupFS(TorchModelV2, nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.

    Adapted from open source code by Alex Nichol:
    https://github.com/unixpickle/obs-tower2
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        f, h, w, c = obs_space.shape
        depth_in = c*f

        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidual(depth_out, 8),
                FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            FixupResidual(depth_in, 8),
            FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers)
        self.hidden_fc = nn.Linear(
            in_features=4096, out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        b, f, h, w, c = x.shape
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 1, 4, 2, 3).reshape(b, f*c, h, w).contiguous()
        x = self.conv_layers(x)
        x = nn.functional.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)

        logits = self.logits_fc(x)
        value = self.value_fc(x)

        self._value = value.squeeze(1)
        return logits, state

    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value


class FixupResidual(nn.Module):
    def __init__(self, depth, num_residual):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()
        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        x = nn.functional.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = nn.functional.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x


ModelCatalog.register_custom_model("fixup_fs", FixupFS)
