# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch


def export_policy_as_jit(policy: torch.nn.Module, normalizer: object | None, path: str, filename="policy.pt"):
    """Export policy into a Torch JIT file optimized for single environment deployment.

    The exported policy is optimized for single environment/robot deployment with batch_size=1.
    For recurrent policies, memory states are pre-initialized and no reset_memory function is included.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, the exporter will try to find one in the policy.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    # Always use the export-friendly wrapper if available.
    if hasattr(policy, "prepare_for_export"):
        policy = policy.prepare_for_export()
        # If no external normalizer is provided, or if it's just Identity, try to get the actual encoder from the policy
        if normalizer is None or isinstance(normalizer, torch.nn.Identity):
            actual_encoder = getattr(policy, "encoder", None)
            if actual_encoder is not None:
                normalizer = actual_encoder

    is_recurrent = getattr(policy, "is_recurrent", False)

    if not is_recurrent:
        exporter = _FlatPolicyExporter(policy, normalizer)
    else:
        # Determine RNN type
        if hasattr(policy, "memory_a"):
            rnn_mod = policy.memory_a.rnn
        elif hasattr(policy, "actor") and hasattr(policy.actor, "memory"):
            rnn_mod = policy.actor.memory.rnn
        else:
            raise ValueError("Cannot locate RNN module in the prepared policy for export.")

        if isinstance(rnn_mod, torch.nn.LSTM):
            exporter = _LSTMPolicyExporter(policy, normalizer)
        else:  # GRU
            exporter = _GRUPolicyExporter(policy, normalizer)

    exporter.export(path, filename)


def export_policy_as_onnx(
    policy: torch.nn.Module, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file optimized for single environment deployment.

    The exported policy is optimized for single environment/robot deployment with batch_size=1.
    Dynamic batch size axes are removed for better performance and compatibility.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, the exporter will try to find one in the policy.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    # If the policy has a dedicated export preparation method, use it.
    if hasattr(policy, "prepare_for_export"):
        policy = policy.prepare_for_export()
        # If no external normalizer is provided, or if it's just Identity, try to get the actual encoder from the policy
        if normalizer is None or isinstance(normalizer, torch.nn.Identity):
            actual_encoder = getattr(policy, "encoder", None)
            if actual_encoder is not None:
                normalizer = actual_encoder

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    exporter = _PolicyExporter(policy, normalizer, export_onnx=True, verbose=verbose)
    exporter.export(path, filename)


class _BasePolicyExporter(torch.nn.Module):
    """Base class for policy exporters."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        # Copy normalizer/encoder
        if normalizer:
            self.encoder = copy.deepcopy(normalizer)
        else:
            self.encoder = torch.nn.Identity()

        # Copy policy components
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self.actor, "_forward_dict"):
                self.actor.forward = self.actor._forward_flat

            # Handle recurrent setup
            if hasattr(policy, "memory_a"):
                self.rnn = copy.deepcopy(policy.memory_a.rnn)

            # Handle noise prediction
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                self._fix_noise_prediction(policy)

        elif hasattr(policy, "student"):  # For distillation
            self.actor = copy.deepcopy(policy.student)
            if hasattr(policy, "memory_s"):
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                self._fix_noise_prediction(policy)
        else:
            raise ValueError("Policy does not have an actor/student module.")

    def _fix_noise_prediction(self, policy):
        """Fix the output layer for noise prediction policies."""
        last_layer = self.actor.layers[-1]
        if isinstance(last_layer, torch.nn.Linear):
            num_actions = policy.num_actions
            new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
            new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
            if last_layer.bias is not None:
                new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
            self.actor.layers[-1] = new_last_layer

    def export(self, path, filename):
        """Export the policy to JIT format."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filepath)


class _FlatPolicyExporter(_BasePolicyExporter):
    """Exporter for non-recurrent policies."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the policy."""
        x_encoded = self.encoder(x)
        if x_encoded.dim() == 1:
            x_encoded = x_encoded.unsqueeze(0)
        return self.actor(x_encoded)


class _LSTMPolicyExporter(_BasePolicyExporter):
    """Exporter for LSTM-based policies."""

    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)
        # Initialize hidden states for single environment (batch_size=1)
        device = next(policy.parameters()).device
        self.hidden_state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
        self.cell_state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM policy."""
        x_encoded = self.encoder(x)
        if x_encoded.dim() == 1:
            x_encoded = x_encoded.unsqueeze(0)

        x_seq = x_encoded.unsqueeze(0)  # Add sequence dimension
        x_out, (h_out, c_out) = self.rnn(x_seq, (self.hidden_state, self.cell_state))
        self.hidden_state = h_out
        self.cell_state = c_out
        return self.actor(x_out.squeeze(0))


class _GRUPolicyExporter(_BasePolicyExporter):
    """Exporter for GRU-based policies."""

    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)
        # Initialize hidden state for single environment (batch_size=1)
        device = next(policy.parameters()).device
        self.hidden_state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU policy."""
        x_encoded = self.encoder(x)
        if x_encoded.dim() == 1:
            x_encoded = x_encoded.unsqueeze(0)

        x_seq = x_encoded.unsqueeze(0)  # Add sequence dimension
        x_out, h_out = self.rnn(x_seq, self.hidden_state)
        self.hidden_state = h_out
        return self.actor(x_out.squeeze(0))


class _PolicyExporter(torch.nn.Module):
    """ONNX-only policy exporter."""

    def __init__(self, policy, normalizer=None, export_onnx=False, verbose=False):
        super().__init__()
        self.is_recurrent = getattr(policy, "is_recurrent", False)
        self.export_onnx = export_onnx
        self.verbose = verbose

        # Copy normalizer/encoder
        if normalizer:
            self.encoder = copy.deepcopy(normalizer)
        else:
            self.encoder = torch.nn.Identity()

        # Copy policy components
        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if hasattr(self.actor, "_forward_dict"):
                self.actor.forward = self.actor._forward_flat

            # Handle recurrent setup
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
                self._setup_recurrent_memory()

            # Handle noise prediction
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                self._fix_noise_prediction(policy)

        elif hasattr(policy, "student"):  # For distillation
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
                self._setup_recurrent_memory()
            if hasattr(policy, "noise_std_type") and policy.noise_std_type == "pred":
                self._fix_noise_prediction(policy)
        else:
            raise ValueError("Policy does not have an actor/student module.")

    def _setup_recurrent_memory(self):
        """Initialize memory states for single environment (batch_size=1)."""
        device = next(self.parameters()).device
        if isinstance(self.rnn, torch.nn.LSTM):
            self.hidden_state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
            self.cell_state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
            self.is_lstm = True
        else:  # GRU
            self.hidden_state = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size, device=device)
            self.is_lstm = False

    def _fix_noise_prediction(self, policy):
        """Fix the output layer for noise prediction policies."""
        last_layer = self.actor.layers[-1]
        if isinstance(last_layer, torch.nn.Linear):
            num_actions = policy.num_actions
            new_last_layer = torch.nn.Linear(last_layer.in_features, num_actions, bias=True)
            new_last_layer.weight.data.copy_(last_layer.weight.data[:num_actions, :])
            if last_layer.bias is not None:
                new_last_layer.bias.data.copy_(last_layer.bias.data[:num_actions])
            self.actor.layers[-1] = new_last_layer

    def forward(self, x: torch.Tensor, h_in=None, c_in=None):
        """Forward pass through the policy."""
        # Apply encoder/normalizer
        x_encoded = self.encoder(x)

        # Ensure batch dimension
        if x_encoded.dim() == 1:
            x_encoded = x_encoded.unsqueeze(0)

        if self.is_recurrent:
            return self._forward_recurrent_onnx(x_encoded, h_in, c_in)
        else:
            # Non-recurrent case
            return self.actor(x_encoded)

    def _forward_recurrent_onnx(self, x_encoded: torch.Tensor, h_in=None, c_in=None):
        """ONNX-friendly forward pass for recurrent networks."""
        x_seq = x_encoded.unsqueeze(0)  # Add sequence dimension
        if c_in is not None:
            # LSTM case
            x_out, (h_out, c_out) = self.rnn(x_seq, (h_in, c_in))
            return self.actor(x_out.squeeze(0)), h_out, c_out
        else:
            # GRU case
            x_out, h_out = self.rnn(x_seq, h_in)
            return self.actor(x_out.squeeze(0)), h_out

    def export(self, path, filename):
        """Export the policy to ONNX format."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        self.to("cpu")
        self._export_onnx(filepath)

    def _export_onnx(self, filepath):
        """Export as ONNX file."""
        if self.is_recurrent:
            # Determine input dimension
            input_dim = self._get_input_dimension()
            obs = torch.zeros(1, input_dim)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.is_lstm:
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    filepath,
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            else:  # GRU
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    filepath,
                    export_params=True,
                    opset_version=11,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
        else:
            # Non-recurrent case
            obs = torch.zeros(1, self.actor.layers[0].in_features)
            torch.onnx.export(
                self,
                (obs,),
                filepath,
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )

    def _get_input_dimension(self):
        """Get the input dimension for the policy."""
        if isinstance(self.encoder, torch.nn.Identity):
            return self.rnn.input_size
        elif hasattr(self.encoder, "input_dim"):
            return self.encoder.input_dim
        elif hasattr(self.encoder, "layers") and isinstance(self.encoder.layers[0], torch.nn.Linear):
            return self.encoder.layers[0].in_features
        else:
            raise TypeError(f"Cannot determine input dimension for encoder of type {type(self.encoder)}")
