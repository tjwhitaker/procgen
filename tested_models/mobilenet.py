from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
import math
from random import random

torch, nn = try_import_torch()

# See https://github.com/hoya012/pytorch-MobileNet/blob/master/MobileNet-pytorch.ipynb


class depthwise_conv(nn.Module):
    def __init__(self, nin, kernel_size, padding, bias=False, stride=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        return out


class dw_block(nn.Module):
    def __init__(self, nin, kernel_size, padding=1, bias=False, stride=1):
        super(dw_block, self).__init__()
        self.dw_block = nn.Sequential(
            depthwise_conv(nin, kernel_size, padding, bias, stride),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.dw_block(x)
        return out


class one_by_one_block(nn.Module):
    def __init__(self, nin, nout, padding=1, bias=False, stride=1):
        super(one_by_one_block, self).__init__()
        self.one_by_one_block = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=1, stride=stride,
                      padding=padding, bias=bias),
            nn.ReLU(True)
        )

    def forward(self, x):
        out = self.one_by_one_block(x)
        return out


class MobileNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        _, _, input_channels = obs_space.shape

        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.ReLU(True),

            dw_block(32, kernel_size=3),
            one_by_one_block(32, 64),

            dw_block(64, kernel_size=3, stride=2),
            one_by_one_block(64, 128),

            dw_block(128, kernel_size=3),
            one_by_one_block(128, 128),

            dw_block(128, kernel_size=3, stride=2),
            one_by_one_block(128, 256),

            dw_block(256, kernel_size=3),
            one_by_one_block(256, 256),

            dw_block(256, kernel_size=3, stride=2),
            one_by_one_block(256, 512),

            dw_block(512, kernel_size=3),
            one_by_one_block(512, 512),

            dw_block(512, kernel_size=3, stride=2),
            one_by_one_block(512, 512),
        )

        self.logits_fc = nn.Linear(512, num_outputs)
        self.value_fc = nn.Linear(512, 1)

    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def forward(self, input_dict, state, seq_lens):
        # print(sum(p.numel() for p in self.parameters() if p.requires_grad))

        x = input_dict["obs"].float()
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()

        body_output = self.network(x)

        avg_pool_output = nn.functional.adaptive_avg_pool2d(
            body_output, (1, 1))
        avg_pool_flat = avg_pool_output.view(avg_pool_output.size(0), -1)

        logits = self.logits_fc(avg_pool_flat)
        value = self.value_fc(avg_pool_flat)

        self._value = value.squeeze(1)

        return logits, state

    def create_ensemble(self, population):
        self.ensemble_weights = []
        self.og_weights = self.logits_fc.weight.detach().clone()

        for _ in range(population):
            weights = self.og_weights.detach().clone()

            # Prune weights
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    if random() < 0.2:
                        weights[i, j] = 0

            # Gaussian Noise
            # std = 0.01
            # noise = torch.randn(weights.size()).cuda()*std
            # weights += noise

            self.ensemble_weights.append(weights)

    def ensemble_forward(self, obs, population):
        with torch.no_grad():
            x = torch.from_numpy(obs).to(
                torch.device("cuda")).unsqueeze(0).float()
            x = x / 255.0
            x = x.permute(0, 3, 1, 2)
            x = x.contiguous()
            x = self.conv_layers(x)
            x = nn.functional.relu(x)
            x = x.view(x.shape[0], -1)
            x = self.hidden_fc(x)
            x = nn.functional.relu(x)

            # Prime ensemble with original output
            output = self.logits_fc(x)
            ensemble_logits = [output]

            for weights in self.ensemble_weights:
                self.logits_fc.weight = nn.Parameter(
                    weights, requires_grad=False)

                ensemble_logits.append(self.logits_fc(x))

            logits = sum(ensemble_logits)

            # Reset paramaters to original
            self.logits_fc.weight = nn.Parameter(
                self.og_weights, requires_grad=False)

            dist = torch.distributions.Categorical(logits=logits)

            return dist.sample().cpu()


ModelCatalog.register_custom_model("mobilenet", MobileNet)
