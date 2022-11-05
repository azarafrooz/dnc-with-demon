# Copyright 2017 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import torch
import torch.nn.functional as F
from Visualize.BitmapTask import visualize_bitmap_task
from Utils import Visdom
from Utils import universal as U

import numpy as np


class BitmapTask(torch.utils.data.Dataset):
    def __init__(self):
        super(BitmapTask, self).__init__()

        self._img = Visdom.Image("preview")

    def set_dump_dir(self, dir):
        self._img.set_dump_dir(dir)

    def __len__(self):
        return 0x7FFFFFFF

    def visualize_preview(self, data, net_output):
        img = visualize_bitmap_task(
            data["input"], [data["output"], U.sigmoid(net_output)]
        )
        self._img.draw(img)

    def loss(self, net_output, target):
        return F.binary_cross_entropy_with_logits(
            net_output, target, reduction="sum"
        ) / net_output.size(0)

    def accuracy(self, net_output, target):
        return F.binary_cross_entropy_with_logits(
            net_output, target, reduction="sum"
        ) / net_output.size(0)

    def demon_loss(self, net_output, target, saved_actions, device):
        """
        computes the loss for the demon
        :param net_output:
        :param target:
        :param saved_actions:
        :return:
        """
        net_output = net_output.detach()
        loss =  F.binary_cross_entropy_with_logits(
            net_output, target, reduction="none"
        ).sum(dim=-1)

        policy_losses = [] # list to save actor (policy) loss

        discount_factor = 0.99
        for i in range(0, loss.size(1)):  # computing expected total reward
            discount_vector = torch.from_numpy(np.array([np.power(discount_factor,i) for i in range(loss.size(1)-i)])).to(device)
            policy_losses.append(((saved_actions[i].log_prob).squeeze(1) * (discount_vector*loss[:, i:]).mean(dim=1)))

        demon_loss = torch.stack(policy_losses).mean(dim=0)/loss.size(1)

        return demon_loss

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass
