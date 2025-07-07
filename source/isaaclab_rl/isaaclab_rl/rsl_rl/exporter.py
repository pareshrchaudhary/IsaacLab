# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from typing import Callable, Union


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    # TODO: need to be tested

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = policy.is_recurrent
        self.forward: Callable[..., torch.Tensor]
        self.reset: Callable[[], None]
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self.actor, "_forward_dict"):
                self.actor.forward = self.actor._forward_flat

            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                last_layer = self.actor.layers[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    num_actions = policy.num_actions
                    new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
                    new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
                    if last_layer.bias is not None:
                        new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
                    self.actor.layers[-1] = new_last_layer
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                last_layer = self.actor.layers[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    num_actions = policy.num_actions
                    new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
                    new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
                    if last_layer.bias is not None:
                        new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
                    self.actor.layers[-1] = new_last_layer
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        else:
            self.forward = self.forward_flat
            self.reset = self.reset_flat
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset_flat(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = policy.is_recurrent
        self.forward: Union[Callable[..., torch.Tensor], Callable[..., tuple[torch.Tensor, ...]]]
        # copy policy parameters
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self.actor, "_forward_dict"):
                self.actor.forward = self.actor._forward_flat

            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                last_layer = self.actor.layers[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    num_actions = policy.num_actions
                    new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
                    new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
                    if last_layer.bias is not None:
                        new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
                    self.actor.layers[-1] = new_last_layer
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                last_layer = self.actor.layers[-1]
                if isinstance(last_layer, torch.nn.Linear):
                    num_actions = policy.num_actions
                    new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
                    new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
                    if last_layer.bias is not None:
                        new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
                    self.actor.layers[-1] = new_last_layer
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.forward = self.forward_lstm
        else:
            self.forward = self.forward_flat
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor.layers[0].in_features)
            torch.onnx.export(
                self,
                (obs,),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
