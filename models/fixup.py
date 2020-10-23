from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
import math
from random import random

import kornia

torch, nn = try_import_torch()


class FixupCNN(TorchModelV2, nn.Module):
    """
    A larger version of the IMPALA CNN with Fixup init.
    See Fixup: https://arxiv.org/abs/1901.09321.

    Adapted from open source code by Alex Nichol:
    https://github.com/unixpickle/obs-tower2
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs, model_config, name)
        nn.Module.__init__(self)

        _, _, depth_in = obs_space.shape

        # transforms = [
        #     kornia.augmentation.RandomResizedCrop(
        #         size=(64, 64), scale=(0.75, 1.0)),
        # ]

        # layers = [*transforms]

        layers = []

        for depth_out in [32, 64, 128]:
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

        self.hidden_fc = nn.Linear(in_features=8192, out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous()
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

    def create_ensemble(self, population):
        self.ensemble_weights = []
        self.og_weights = self.logits_fc.weight.detach().clone()

        for _ in range(population):
            weights = self.og_weights.detach().clone()

            # Prune weights
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    if random() < 0.25:
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

            if random() < 0.5:
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

    def brain_damage(self):
        pass


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


ModelCatalog.register_custom_model("fixup", FixupCNN)
