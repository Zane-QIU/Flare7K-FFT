import torch, torch.nn as nn
import torch.nn.functional as F

class FFTFusion(nn.Module):
    """Token-shape (B N C)  ⇆  NCHW 频域抑条纹"""
    def __init__(self, dim):
        super().__init__()
        # 深度可分 1×1，等价在频域对每通道幅值做平滑
        self.attn_amp = nn.Conv2d(dim, dim, 1, 1, groups=dim, bias=False)

    def forward(self, x, H, W):
        """
        x : (B, N, C)  token 格式
        H, W : 当前特征图长宽
        """
        B, N, C = x.shape
        feat = x.transpose(1, 2).reshape(B, C, H, W)        # → B C H W

        # ① FFT
        freq = torch.fft.rfft2(feat, norm='ortho')          # 复数张量
        amp  = torch.abs(freq)                              # 幅值
        phase= torch.angle(freq)                            # 相位

        # ② 幅值卷积抑制尖峰
        amp = self.attn_amp(amp)

        # ③ 还原复数谱并 IFFT
        freq_filtered = torch.polar(amp, phase)
        feat_rec = torch.fft.irfft2(freq_filtered, s=(H, W), norm='ortho')

        # ④ 残差 + reshape 回 token
        feat_out = (feat_rec + feat).reshape(B, C, -1).transpose(1, 2)  # B N C
        return feat_out
