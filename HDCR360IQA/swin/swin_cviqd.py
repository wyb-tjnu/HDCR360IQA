import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from swin.swin_tra import swin_tiny_patch4_window7_224
import math



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, requires_grad=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p, requires_grad=requires_grad)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)
        return x.view(x.size(0), -1)


class RegressionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)



class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        sqrt2 = math.sqrt(2.0)
        self.register_buffer('dec_lo', torch.tensor([1, 1], dtype=torch.float32).view(1, 1, 1, 2) / sqrt2)
        self.register_buffer('dec_hi', torch.tensor([-1, 1], dtype=torch.float32).view(1, 1, 1, 2) / sqrt2)

    def forward(self, x):
        B, C, H, W = x.shape
        dec_lo = self.dec_lo
        dec_hi = self.dec_hi
        x_lo = F.conv2d(x, dec_lo.repeat(C, 1, 1, 1), stride=(1, 2), padding=(0, 0), groups=C)
        x_hi = F.conv2d(x, dec_hi.repeat(C, 1, 1, 1), stride=(1, 2), padding=(0, 0), groups=C)
        LL = F.conv2d(x_lo, dec_lo.transpose(2, 3).repeat(C, 1, 1, 1), stride=(2, 1), padding=(0, 0), groups=C)
        LH = F.conv2d(x_lo, dec_hi.transpose(2, 3).repeat(C, 1, 1, 1), stride=(2, 1), padding=(0, 0), groups=C)
        HL = F.conv2d(x_hi, dec_lo.transpose(2, 3).repeat(C, 1, 1, 1), stride=(2, 1), padding=(0, 0), groups=C)
        HH = F.conv2d(x_hi, dec_hi.transpose(2, 3).repeat(C, 1, 1, 1), stride=(2, 1), padding=(0, 0), groups=C)
        return torch.cat((LL, LH, HL, HH), dim=1)


class WaveletEnhance(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.dwt = DWT()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c * 4, in_c, kernel_size=1, bias=False),
            nn.GELU()
        )
        self.alpha = nn.Parameter(torch.tensor([0.3]))

    def forward(self, x):
        w = self.dwt(x)
        w = self.conv(w)
        w = F.interpolate(w, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.alpha * w


class DistortionAwareFeatureModulation(nn.Module):
    def __init__(self, channels, stage_type="shallow"):
        super().__init__()
        self.stage_type = stage_type
        self.spatial_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.residual_conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        dist_var = (x - mean) ** 2
        spatial_weight = self.sigmoid(self.act(self.spatial_conv(dist_var)))
        channel_weight = self.channel_attn(x)
        dist_weight = spatial_weight * channel_weight
        if self.stage_type == "deep":
            dynamic_gain = 1.0 + self.gamma * 1.25 * dist_weight
        else:
            dynamic_gain = 1.0 + self.gamma * dist_weight
        out = x * dynamic_gain + self.residual_conv(x)
        return out



class HDCR360iq(nn.Module):
    def __init__(self, weight_path="/public/home/huu147/mamba/swin/swin_tiny_patch4_window7_224.pth",
                 reg_hidden_dim=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.backbone = swin_tiny_patch4_window7_224(pretrained=False).to(self.device)


        if weight_path and os.path.isfile(weight_path):
            print(f"===== 正在加载Swin预训练权重: {weight_path} =====")
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"【权重加载校验】缺失参数数量: {len(missing_keys)}")
            if len(missing_keys) > 0:
                print("前15个缺失权重key:", missing_keys[:15])
            print(f"【权重加载校验】多余参数数量: {len(unexpected_keys)}")
            if len(unexpected_keys) > 0:
                print("前15个多余权重key:", unexpected_keys[:15])
            print("==============================================\n")
        else:
            print(f"警告：权重文件不存在，路径 {weight_path}")


        self.gem1 = GeM(requires_grad=True)

        self.dafm1 = DistortionAwareFeatureModulation(96, stage_type="shallow")
        self.dafm2 = DistortionAwareFeatureModulation(192, stage_type="shallow")
        self.dafm3 = DistortionAwareFeatureModulation(384, stage_type="deep")
        self.dafm4 = DistortionAwareFeatureModulation(768, stage_type="deep")

        self.wave1 = WaveletEnhance(96)
        self.wave2 = WaveletEnhance(192)

        self.main_reduce = nn.Linear(1440, 768, bias=False)
        self.reduce = nn.Linear(1536, 1536, bias=False)
        self.viewport_score_head = RegressionHead(1536, reg_hidden_dim)
        self.to(self.device)


    def seq2feature_map(self, seq):
        b, n, c = seq.shape
        side = int(n ** 0.5)
        return seq.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        x = x.to(self.device)
        B, M, N, C, H, W = x.shape
        imgs = x.view(B * M * N, C, H, W)

        four_stage_seq = self.backbone(imgs)


        for idx, seq in enumerate(four_stage_seq):
            b, n, c = seq.shape
            s = int(n ** 0.5)

        s0_seq, s1_seq, s2_seq, s3_seq = four_stage_seq
        base1 = self.seq2feature_map(s0_seq)
        base2 = self.seq2feature_map(s1_seq)
        base3 = self.seq2feature_map(s2_seq)
        focus2 = self.seq2feature_map(s3_seq)

        base1 = self.dafm1(base1)
        base2 = self.dafm2(base2)
        base3 = self.dafm3(base3)
        focus3 = self.dafm4(focus2)

        base1 = self.wave1(base1)
        base2 = self.wave2(base2)

        p1 = self.gem1(base1)
        p2 = self.gem1(base2)
        p3 = self.gem1(base3)
        p4 = self.gem1(focus3)

        main = torch.cat([p1, p2, p3, p4], dim=1)
        main = self.main_reduce(main)

        guidance = self.gem1(focus3)

        fused = torch.cat([main, guidance], dim=1)
        fused = self.reduce(fused)

        viewport_scores = self.viewport_score_head(fused).view(B, M, N)
        final_score = viewport_scores.mean(dim=[1, 2])

        return final_score