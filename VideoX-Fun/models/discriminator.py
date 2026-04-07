import torch
import torch.nn as nn
import torch.nn.functional as F


class CondFeatureDiscHead(nn.Module):
    def __init__(
        self,
        in_channels=3840,
        cond_dim=3840,
        hidden_channels=512,
        time_hidden_dim=256,
    ):
        super().__init__()

        # Turn scalar t into a vector we can inject into conv features
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, hidden_channels),
        )

        # Turn pooled caption embedding into a vector we can inject too
        self.caption_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Main conv net
        self.conv_in = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

        self.conv_out = nn.Conv2d(
            hidden_channels // 2,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, t, caption_pooled):
        """
        x: [B, C, H, W]
        t: [B] or [B, 1]
        caption_pooled: [B, cond_dim]
        returns: [B, num_logits]
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # [B, 1]

        # Force conditioning path to fp32 for stability
        t = t.float()
        caption_pooled = caption_pooled.float()

        # Main image feature path
        h = self.conv_in(x)

        # Build conditioning vectors
        t_embed = self.time_mlp(t)[:, :, None, None]              # [B, hidden, 1, 1]
        c_embed = self.caption_mlp(caption_pooled)[:, :, None, None]

        # Inject conditioning additively
        h = h + t_embed + c_embed

        # More convs
        h = self.conv_mid(h)

        # Patch logits
        logits = self.conv_out(h)   # [B, 1, H', W']
        logits = logits.reshape(logits.shape[0], -1)

        return logits

class MultiFeatureDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels_list,
        cond_dim=3840,
        hidden_channels=512,
        time_hidden_dim=256,
    ):
        super().__init__()

        self.feature_heads = nn.ModuleList([
            CondFeatureDiscHead(
                in_channels=in_ch,
                cond_dim=cond_dim,
                hidden_channels=hidden_channels,
                time_hidden_dim=time_hidden_dim,
            )
            for in_ch in in_channels_list
        ])

    def forward(self, feature_maps, t, caption_pooled):
        """
        feature_maps: list of [B, C, H, W]
        t: [B] or [B, 1]
        caption_pooled: [B, cond_dim]
        returns: [B, total_num_logits]
        """
        assert len(feature_maps) == len(self.feature_heads), (
            f"Expected {len(self.feature_heads)} feature maps, got {len(feature_maps)}"
        )

        outs = []
        for fmap, head in zip(feature_maps, self.feature_heads):
            outs.append(head(fmap, t, caption_pooled))

        return torch.cat(outs, dim=1)