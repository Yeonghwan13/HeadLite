"""HeadLite convolutional branches, metadata encoder and prediction heads."""
import torch
from torch import nn

META_ENCODE_DIM = 32
META_FEATURE_DIM = 6

class MetaEncoder(nn.Module):


    def __init__(self, input_dim: int = META_FEATURE_DIM,
                 hidden_dim: int = META_ENCODE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)

class Encoder40ChMeta(nn.Module):






    def __init__(
        self,
        acc_gyr_dim1: int = 32,
        prs_dim1: int = 128,
        dense_dim: int = 256,
        dropout_rate: float = 0.5,
        meta_feature_dim: int = META_FEATURE_DIM,
        meta_encode_dim: int = META_ENCODE_DIM,
        proj_dim: int = 128,
    ) -> None:
        super().__init__()

        self.conv1d_acc = nn.Sequential(
            nn.Conv1d(3, acc_gyr_dim1, 30),
            nn.BatchNorm1d(acc_gyr_dim1), nn.ReLU(inplace=True),
            nn.Conv1d(acc_gyr_dim1, acc_gyr_dim1 * 2, 30),
            nn.BatchNorm1d(acc_gyr_dim1 * 2), nn.ReLU(inplace=True),
            nn.Conv1d(acc_gyr_dim1 * 2, acc_gyr_dim1 * 4, 30),
            nn.BatchNorm1d(acc_gyr_dim1 * 4), nn.ReLU(inplace=True),
        )
        self.conv1d_gyr = nn.Sequential(
            nn.Conv1d(3, acc_gyr_dim1, 30),
            nn.BatchNorm1d(acc_gyr_dim1), nn.ReLU(inplace=True),
            nn.Conv1d(acc_gyr_dim1, acc_gyr_dim1 * 2, 30),
            nn.BatchNorm1d(acc_gyr_dim1 * 2), nn.ReLU(inplace=True),
            nn.Conv1d(acc_gyr_dim1 * 2, acc_gyr_dim1 * 4, 30),
            nn.BatchNorm1d(acc_gyr_dim1 * 4), nn.ReLU(inplace=True),
        )
        self.conv1d_prs = nn.Sequential(
            nn.Conv1d(40, prs_dim1, 30),
            nn.BatchNorm1d(prs_dim1), nn.ReLU(inplace=True),
            nn.Conv1d(prs_dim1, prs_dim1 * 2, 30),
            nn.BatchNorm1d(prs_dim1 * 2), nn.ReLU(inplace=True),
            nn.Conv1d(prs_dim1 * 2, prs_dim1 * 4, 30),
            nn.BatchNorm1d(prs_dim1 * 4), nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        acc_dim = acc_gyr_dim1 * 4   # 128
        gyr_dim = acc_gyr_dim1 * 4   # 128
        prs_dim = prs_dim1 * 4       # paper config (prs_dim1=48): 192
        self.proj_dim = proj_dim

        self.acc_proj = nn.Sequential(
            nn.Linear(acc_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.gyr_proj = nn.Sequential(
            nn.Linear(gyr_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.prs_proj = nn.Sequential(
            nn.Linear(prs_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.meta_encoder = MetaEncoder(meta_feature_dim, meta_encode_dim)
        self.combined_dim = proj_dim * 3 + meta_encode_dim  # paper config: 128*3 + 64 = 448

        half_dim = dense_dim // 2

        self.dense_mean = nn.Sequential(
            nn.Linear(self.combined_dim, dense_dim),
            nn.BatchNorm1d(dense_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(dense_dim, half_dim),
            nn.BatchNorm1d(half_dim), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(half_dim, 1),
        )

        self.dense_var = nn.Sequential(
            nn.Linear(self.combined_dim, dense_dim),
            nn.BatchNorm1d(dense_dim), nn.ReLU(inplace=True),
            nn.Linear(dense_dim, half_dim),
            nn.BatchNorm1d(half_dim), nn.ReLU(inplace=True),
            nn.Linear(half_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        inputs_acc: torch.Tensor,
        inputs_gyr: torch.Tensor,
        inputs_prs: torch.Tensor,
        meta: torch.Tensor,
        return_embedding: bool = False,
    ):
        acc_f = self.gap(self.conv1d_acc(inputs_acc)).squeeze(-1)
        gyr_f = self.gap(self.conv1d_gyr(inputs_gyr)).squeeze(-1)
        prs_f = self.gap(self.conv1d_prs(inputs_prs)).squeeze(-1)

        acc_p = self.acc_proj(acc_f)
        gyr_p = self.gyr_proj(gyr_f)
        prs_p = self.prs_proj(prs_f)
        meta_features = self.meta_encoder(meta)

        combined = torch.cat([acc_p, gyr_p, prs_p, meta_features], dim=1)

        mean = self.dense_mean(combined)
        var = self.dense_var(combined)

        if return_embedding:
            return mean, var, combined

        return mean, var
