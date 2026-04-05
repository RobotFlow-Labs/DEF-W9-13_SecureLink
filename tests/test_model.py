"""Smoke tests for SecureLink model architecture."""

from __future__ import annotations

import pytest
import torch

from securelink.model import (
    CNN1DBlock,
    MultiHeadAttentionFusion,
    SecureLinkModel,
    UnimodalBranch,
    build_model,
)
from securelink.utils import ModelConfig


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        cnn_filters=64,
        cnn_kernel_sizes=[3, 2],
        lstm_hidden=64,
        lstm_layers=1,
        attention_heads=4,
        attention_layers=2,
        embedding_dim=256,
        ff_dim=512,
        dropout=0.1,
    )


@pytest.fixture
def small_config() -> ModelConfig:
    """Smaller config for faster tests."""
    return ModelConfig(
        cnn_filters=16,
        cnn_kernel_sizes=[3, 2],
        lstm_hidden=16,
        lstm_layers=1,
        attention_heads=2,
        attention_layers=1,
        embedding_dim=32,
        ff_dim=64,
        dropout=0.0,
    )


class TestCNN1DBlock:
    def test_output_shape(self):
        block = CNN1DBlock(in_channels=52, filters=64, kernel_sizes=(3, 2))
        x = torch.randn(4, 6, 52)  # (batch, seq, features)
        out = block(x)
        assert out.shape[0] == 4
        assert out.shape[2] == 64
        assert out.shape[1] > 0  # reduced sequence length

    def test_different_input_sizes(self):
        for in_ch in [8, 26, 52]:
            block = CNN1DBlock(in_channels=in_ch, filters=32)
            x = torch.randn(2, 10, in_ch)
            out = block(x)
            assert out.shape[0] == 2
            assert out.shape[2] == 32


class TestUnimodalBranch:
    def test_output_shape(self):
        branch = UnimodalBranch(in_features=52, cnn_filters=64, lstm_hidden=64)
        x = torch.randn(4, 6, 52)
        out = branch(x)
        assert out.shape[0] == 4
        assert out.shape[2] == 128  # bidirectional: 64 * 2

    def test_mems_branch(self):
        branch = UnimodalBranch(in_features=8, cnn_filters=64, lstm_hidden=64)
        x = torch.randn(4, 6, 8)
        out = branch(x)
        assert out.shape[0] == 4
        assert out.shape[2] == 128


class TestAttentionFusion:
    def test_output_shape(self):
        fusion = MultiHeadAttentionFusion(
            embed_dim=256, num_heads=4, num_layers=2, ff_dim=512
        )
        x = torch.randn(4, 3, 256)
        out = fusion(x)
        assert out.shape == x.shape

    def test_single_layer(self):
        fusion = MultiHeadAttentionFusion(
            embed_dim=64, num_heads=2, num_layers=1, ff_dim=128
        )
        x = torch.randn(2, 5, 64)
        out = fusion(x)
        assert out.shape == (2, 5, 64)


class TestSecureLinkModel:
    def test_forward_pass(self, config: ModelConfig):
        model = build_model(config, csi_features=52, mems_features=8)
        csi = torch.randn(4, 6, 52)
        mems = torch.randn(4, 6, 8)
        emb = model(csi, mems)
        assert emb.shape == (4, 256)

    def test_l2_normalized(self, config: ModelConfig):
        model = build_model(config, csi_features=52, mems_features=8)
        csi = torch.randn(4, 6, 52)
        mems = torch.randn(4, 6, 8)
        emb = model(csi, mems)
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_parameter_count(self, config: ModelConfig):
        model = build_model(config, csi_features=52, mems_features=8)
        params = model.count_parameters()
        assert params > 0
        print(f"Parameter count: {params:,}")

    def test_gradient_flow(self, small_config: ModelConfig):
        model = build_model(small_config, csi_features=52, mems_features=8)
        csi = torch.randn(2, 6, 52)
        mems = torch.randn(2, 6, 8)
        emb = model(csi, mems)
        loss = emb.sum()
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_csi_features(self, small_config: ModelConfig):
        for k in [26, 52, 64]:
            model = build_model(small_config, csi_features=k, mems_features=8)
            csi = torch.randn(2, 6, k)
            mems = torch.randn(2, 6, 8)
            emb = model(csi, mems)
            assert emb.shape == (2, small_config.embedding_dim)


class TestBuildModel:
    def test_factory_default(self):
        config = ModelConfig()
        model = build_model(config)
        assert isinstance(model, SecureLinkModel)

    def test_factory_custom(self):
        config = ModelConfig(embedding_dim=128)
        model = build_model(config, csi_features=26, mems_features=8)
        csi = torch.randn(1, 6, 26)
        mems = torch.randn(1, 6, 8)
        emb = model(csi, mems)
        assert emb.shape == (1, 128)
