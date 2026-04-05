"""SecureLink model: two-branch 1D-CNN + BiLSTM with attention-based multimodal fusion.

Architecture from paper:
  - Branch 1 (CSI): Conv1D(K, 64, k=3) -> BN -> ReLU -> Pool(2)
                     -> Conv1D(64, 64, k=2) -> BN -> ReLU -> BiLSTM(128) -> x (128-dim)
  - Branch 2 (MEMS): Same architecture with 8 input features
  - Fusion: Concatenate(x, y) -> 2x MultiHeadAttention(4 heads) -> Flatten -> D
  - Embedding: Linear(D, 256) -> L2 normalize -> g (256-dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from securelink.utils import ModelConfig


class CNN1DBlock(nn.Module):
    """Two-layer 1D CNN feature extractor.

    Layer 1: Conv1D(in_ch, 64, kernel=3) + BN + ReLU + MaxPool(2)
    Layer 2: Conv1D(64, 64, kernel=2) + BN + ReLU
    """

    def __init__(
        self,
        in_channels: int,
        filters: int = 64,
        kernel_sizes: tuple[int, int] = (3, 2),
    ):
        super().__init__()
        k1, k2 = kernel_sizes

        self.conv1 = nn.Conv1d(in_channels, filters, kernel_size=k1, padding=k1 // 2)
        self.bn1 = nn.BatchNorm1d(filters)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(filters, filters, kernel_size=k2, padding=0)
        self.bn2 = nn.BatchNorm1d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, features) -- time-major input

        Returns:
            (batch, reduced_seq, 64) -- feature maps
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # Back to (batch, seq, channels)
        return x.transpose(1, 2)


class UnimodalBranch(nn.Module):
    """Single modality feature extractor: 1D-CNN + BiLSTM.

    Produces a 128-dim feature vector per sample from a sequence of fingerprint frames.
    """

    def __init__(
        self,
        in_features: int,
        cnn_filters: int = 64,
        cnn_kernel_sizes: tuple[int, int] = (3, 2),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = CNN1DBlock(in_features, cnn_filters, cnn_kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.output_dim = lstm_hidden * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, in_features)

        Returns:
            (batch, seq_out, 128) -- BiLSTM output sequence
        """
        cnn_out = self.cnn(x)  # (batch, reduced_seq, cnn_filters)
        lstm_out, _ = self.lstm(cnn_out)  # (batch, reduced_seq, 128)
        return lstm_out


class AttentionFusionLayer(nn.Module):
    """Single transformer-style self-attention block.

    MultiHeadAttention + residual + LayerNorm + FeedForward + residual + LayerNorm
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention with residual connections.

        Args:
            x: (batch, seq, embed_dim)

        Returns:
            (batch, seq, embed_dim)
        """
        # Self-attention + residual + norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward + residual + norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class MultiHeadAttentionFusion(nn.Module):
    """Multimodal feature fusion via stacked self-attention layers.

    Paper uses 2 attention layers with 4 heads each.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionFusionLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stacked attention layers.

        Args:
            x: (batch, seq, embed_dim) -- concatenated feature map U

        Returns:
            (batch, seq, embed_dim) -- fused features
        """
        for layer in self.layers:
            x = layer(x)
        return x


class SecureLinkModel(nn.Module):
    """Full SecureLink model: two-branch extraction + attention fusion + embedding.

    Architecture:
        CSI input  -> UnimodalBranch -> x (128-dim per timestep)
        MEMS input -> UnimodalBranch -> y (128-dim per timestep)
        Concatenate(x, y) -> U (256-dim per timestep)
        MultiHeadAttentionFusion(U) -> D
        Flatten + Linear(256) + L2 normalize -> g (256-dim embedding)
    """

    def __init__(self, config: ModelConfig, csi_features: int = 52, mems_features: int = 8):
        super().__init__()
        self.config = config

        kernel_sizes = tuple(config.cnn_kernel_sizes)

        # Two-branch unimodal feature extraction
        self.csi_branch = UnimodalBranch(
            in_features=csi_features,
            cnn_filters=config.cnn_filters,
            cnn_kernel_sizes=kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )
        self.mems_branch = UnimodalBranch(
            in_features=mems_features,
            cnn_filters=config.cnn_filters,
            cnn_kernel_sizes=kernel_sizes,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )

        # Feature dimension after concatenation
        fused_dim = self.csi_branch.output_dim + self.mems_branch.output_dim  # 256

        # Multimodal attention fusion
        self.fusion = MultiHeadAttentionFusion(
            embed_dim=fused_dim,
            num_heads=config.attention_heads,
            num_layers=config.attention_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
        )

        # Embedding layer: compress fused features to embedding_dim
        self.embedding = nn.Linear(fused_dim, config.embedding_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d | nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, csi: torch.Tensor, mems: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: extract, fuse, and embed fingerprints.

        Args:
            csi: (batch, M, K) -- CSI phase error sequences
            mems: (batch, M, 8) -- MEMS telemetry sequences

        Returns:
            (batch, embedding_dim) -- L2-normalized embeddings
        """
        # Unimodal feature extraction
        x_csi = self.csi_branch(csi)    # (batch, seq_out, 128)
        x_mems = self.mems_branch(mems)  # (batch, seq_out, 128)

        # Ensure same sequence length (take min if they differ due to pooling)
        min_len = min(x_csi.size(1), x_mems.size(1))
        x_csi = x_csi[:, :min_len, :]
        x_mems = x_mems[:, :min_len, :]

        # Concatenate along feature dimension -> feature map U
        u = torch.cat([x_csi, x_mems], dim=-1)  # (batch, seq_out, 256)

        # Multimodal attention fusion
        d = self.fusion(u)  # (batch, seq_out, 256)

        # Global average pooling over sequence -> (batch, 256)
        d = d.mean(dim=1)

        # Embedding projection + L2 normalization
        g = self.embedding(d)  # (batch, embedding_dim)
        g = F.normalize(g, p=2, dim=-1)

        return g

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    config: ModelConfig, csi_features: int = 52, mems_features: int = 8,
) -> SecureLinkModel:
    """Factory function to build the SecureLink model.

    Args:
        config: Model configuration
        csi_features: Number of CSI input features (subcarriers)
        mems_features: Number of MEMS input features

    Returns:
        SecureLinkModel instance
    """
    return SecureLinkModel(config, csi_features=csi_features, mems_features=mems_features)
