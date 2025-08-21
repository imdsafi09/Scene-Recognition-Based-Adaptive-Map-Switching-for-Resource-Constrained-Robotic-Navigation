# model.py
import math, copy, torch
import torch.nn as nn
import torch.nn.functional as F

try:
    SiLU = nn.SiLU
except AttributeError:
    class SiLU(nn.Module):
        def __init__(self, inplace: bool = False): super().__init__(); self.inplace = inplace
        def forward(self, x): return F.silu(x, inplace=self.inplace)

class SqueezeExcite(nn.Module):
    def __init__(self, c, se_ratio=0.25):
        super().__init__()
        hidden = max(8, int(c * se_ratio))
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c, 1)
    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    def __init__(self, cin, cout, stride=1, expand=4, se_ratio=0.25, drop=0.0):
        super().__init__()
        hidden = cin * expand
        self.stride = stride
        self.use_res = (stride == 1 and cin == cout)
        self.drop = drop
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), SiLU(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), SiLU(inplace=True),
            SqueezeExcite(hidden, se_ratio),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            if self.drop > 0 and self.training:
                out = out * (1 - self.drop)
            out = out + x
        return out

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=160, num_heads=4, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), SiLU(), nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.drop = drop
    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        t = self.norm1(t)
        a, _ = self.attn(t, t, t, need_weights=False)
        if self.drop > 0 and self.training: a = F.dropout(a, p=self.drop)
        t = t + a
        u = self.norm2(t)
        u = self.mlp(u)
        if self.drop > 0 and self.training: u = F.dropout(u, p=self.drop)
        t = t + u
        return t.transpose(1, 2).reshape(B, C, H, W)

class SwiftSceneNet(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, drop_rate: float = 0.1, attn_pool: int = 14):
        super().__init__()
        def C(ch): return int(math.ceil(ch * width_mult / 8) * 8)

        self.stem = nn.Sequential(
            nn.Conv2d(3, C(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C(32)), SiLU(inplace=True)
        )

        cfg = [
            (C(32),  C(32),  1, 1),
            (C(32),  C(48),  2, 2),
            (C(48),  C(80),  2, 2),
            (C(80),  C(128), 2, 3),
        ]
        blocks = []
        for cin, cout, stride, n in cfg:
            blocks.append(MBConv(cin, cout, stride=stride, expand=4, se_ratio=0.25, drop=0.0))
            for _ in range(n - 1):
                blocks.append(MBConv(cout, cout, stride=1, expand=4, se_ratio=0.25, drop=0.0))
        self.blocks = nn.Sequential(*blocks)

        self.attn_conv = nn.Conv2d(C(128), C(160), 1, bias=False)
        self.attn_bn   = nn.BatchNorm2d(C(160))
        self.attn_act  = SiLU(inplace=True)
        self.token_pool = nn.AdaptiveAvgPool2d((attn_pool, attn_pool))
        self.trans = TinyTransformerBlock(dim=C(160), num_heads=4, mlp_ratio=2.0, drop=drop_rate)

        self.head = nn.Sequential(
            nn.Conv2d(C(160), C(192), 1, bias=False),
            nn.BatchNorm2d(C(192)), SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(p=drop_rate),
            nn.Linear(C(192), num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                try: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
                except ValueError: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if getattr(m, "weight", None) is not None: nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.attn_act(self.attn_bn(self.attn_conv(x)))
        x = self.token_pool(x)
        x = self.trans(x)
        x = self.head(x)
        return x

def set_bn_momentum(model: nn.Module, mom: float):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = mom

# Lightweight MACs/GFLOPs report (same as your monolith)
def _count_layers(m: nn.Module) -> int:
    L = 0
    for mod in m.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.MultiheadAttention)):
            L += 1
    return L

def _mhsa_mlp_flops(L: int, D: int, heads: int = 4, mlp_ratio: float = 2.0) -> float:
    dh = D // heads
    qkv = 3 * L * D * D
    scores = L * L * D
    av = L * L * D
    outp = L * D * D
    mlp = L * (D * (mlp_ratio * D) + (mlp_ratio * D) * D)
    return float(qkv + scores + av + outp + mlp)

@torch.no_grad()
def model_report(model: nn.Module, img_h: int, img_w: int, attn_tokens: int = 14*14):
    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 3, img_h, img_w)
    conv_macs = 0
    linear_macs = 0

    def hook_conv(mod, inp, out):
        nonlocal conv_macs
        if not isinstance(mod, nn.Conv2d): return
        out_h, out_w = out.shape[2], out.shape[3]
        c_out = mod.out_channels
        c_in  = mod.in_channels
        k_h, k_w = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        groups = mod.groups
        macs = out_h * out_w * c_out * (k_h * k_w * (c_in // groups))
        conv_macs += macs

    def hook_linear(mod, inp, out):
        nonlocal linear_macs
        if not isinstance(mod, nn.Linear): return
        linear_macs += mod.in_features * mod.out_features

    hooks = []
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(hook_conv))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(hook_linear))
    m(dummy)
    for h in hooks: h.remove()

    D = 160
    for n, p in m.state_dict().items():
        if "attn.in_proj_weight" in n:
            D = p.shape[-1]; break

    mhsa_mlp = _mhsa_mlp_flops(L=attn_tokens, D=D, heads=4, mlp_ratio=2.0)
    total_macs = conv_macs + linear_macs + mhsa_mlp
    gmacs = total_macs / 1e9
    gflops = 2.0 * total_macs / 1e9
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = _count_layers(model)
    return gmacs, gflops, total_params, train_params, layers

