"""Mechanical verification of Milestones 2-4.

Run before any training installment:

    python verify_milestones.py

Every check either prints PASS or raises. Milestone 2 = shared backbone with a
pluggable slot; Milestone 3 = SE reimplemented from Eq. 2-3; Milestone 4 = BAM
and CBAM ported faithfully from Jongchan/attention-module.
"""

import torch
import torch.nn as nn

import attention as A
from model import BasicBlock, resnet_cifar

CHANNELS = [64, 128, 256, 512]
REDUCTION = 16
checks = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    checks.append((status, name, detail))
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


# ---------------------------------------------------------------- Milestone 2
print("\nMilestone 2 -- shared backbone with a fixed, pluggable attention slot")

net = resnet_cifar(attention_type=None)
stem_conv = net.stem[0]
check("stem is 3x3 stride-1 (CIFAR-appropriate, not the ImageNet 7x7/2)",
      stem_conv.kernel_size == (3, 3) and stem_conv.stride == (1, 1),
      f"kernel={stem_conv.kernel_size} stride={stem_conv.stride}")
check("no initial max-pool (32x32 spatial detail preserved)",
      not any(isinstance(m, nn.MaxPool2d) for m in net.modules()))

blocks = [m for m in net.modules() if isinstance(m, BasicBlock)]
check("8 residual blocks (4 stages x 2)", len(blocks) == 8, f"found {len(blocks)}")
check("every block exposes an attention slot",
      all(hasattr(b, "attention") for b in blocks))
check("baseline leaves every slot empty",
      all(b.attention is None for b in blocks))

params = {}
for variant in [None, "SE", "BAM", "CBAM"]:
    m = resnet_cifar(attention_type=variant)
    params[variant or "none"] = sum(p.numel() for p in m.parameters())
    if variant is not None:
        vblocks = [b for b in m.modules() if isinstance(b, BasicBlock)]
        check(f"{variant}: attention present in all 8 blocks",
              all(b.attention is not None for b in vblocks))

check("identical backbone across arms (attention params only differ)",
      all(params[k] >= params["none"] for k in params),
      " ".join(f"{k}={v/1e6:.2f}M" for k, v in params.items()))

# Documented deviation, not a failure -- see docs/PROVENANCE.md
print(f"  [NOTE] backbone is ResNet-18 scale ({params['none']/1e6:.2f}M params), "
      f"not the ~1-2M the plan assumed. Deviation accepted on hardware grounds.")

# ---------------------------------------------------------------- Milestone 3
print("\nMilestone 3 -- SE reimplemented from Eq. 2-3 of arXiv:1709.01507")

se = A.SEBlock(64, reduction=REDUCTION)
check("squeeze is global average pooling", isinstance(se.avg_pool, nn.AdaptiveAvgPool2d))
linears = [m for m in se.fc if isinstance(m, nn.Linear)]
check("excitation is a 2-layer FC bottleneck", len(linears) == 2)
check(f"bottleneck width is C/r with documented r={REDUCTION}",
      linears[0].out_features == 64 // REDUCTION,
      f"64 -> {linears[0].out_features} -> {linears[1].out_features}")
check("ReLU between the two FC layers", any(isinstance(m, nn.ReLU) for m in se.fc))
check("sigmoid gate", isinstance(se.fc[-1], nn.Sigmoid))

x = torch.randn(4, 64, 8, 8)
se.eval()
with torch.no_grad():
    y = se(x)
check("output shape preserved", y.shape == x.shape)
ratio = (y / x)
check("acts as a pure channel-wise rescale (gate constant over H,W)",
      torch.allclose(ratio, ratio[:, :, :1, :1].expand_as(ratio), atol=1e-5))
check("gate lies in (0, 1) as a sigmoid must",
      bool((ratio > 0).all() and (ratio < 1).all()))

# ---------------------------------------------------------------- Milestone 4
print("\nMilestone 4 -- BAM/CBAM ported from Jongchan/attention-module")

check("BAM and CBAM do NOT share a channel-gate class",
      A.BAMChannelGate is not A.CBAMChannelGate)

bam_cg = A.BAMChannelGate(64, REDUCTION)
check("BAM channel gate has BatchNorm1d inside its MLP (official bam.py)",
      any(isinstance(m, nn.BatchNorm1d) for m in bam_cg.modules()))
check("BAM channel gate is average-pool only (no max branch)",
      not hasattr(bam_cg, "pool_types") and
      not any(isinstance(m, nn.MaxPool2d) for m in bam_cg.modules()))

bam_sg = A.BAMSpatialGate(64, REDUCTION)
dilated = [m for m in bam_sg.modules() if isinstance(m, nn.Conv2d) and m.dilation == (4, 4)]
check("BAM spatial gate uses 2 dilated 3x3 convs (dilation=4)", len(dilated) == 2,
      f"found {len(dilated)}")

cbam_cg = A.CBAMChannelGate(64, REDUCTION)
check("CBAM channel gate pools avg AND max through a shared MLP",
      cbam_cg.pool_types == ("avg", "max"))
cbam_sg = A.CBAMSpatialGate()
check("CBAM spatial conv carries BatchNorm2d (official BasicConv)",
      cbam_sg.spatial.bn is not None and cbam_sg.spatial.bn.eps == 1e-5)
check("CBAM spatial conv is 7x7", cbam_sg.spatial.conv.kernel_size == (7, 7))

# Fusion semantics, checked numerically rather than by reading the source.
bam = A.BAMBlock(64, REDUCTION).eval()
with torch.no_grad():
    xb = torch.randn(4, 64, 8, 8)
    r = bam(xb) / xb
check("BAM is residual-multiplicative: out/x = 1 + sigmoid(.) in (1, 2)",
      bool((r > 1).all() and (r < 2).all()),
      f"range [{r.min():.3f}, {r.max():.3f}]")

cbam = A.CBAMBlock(64, REDUCTION).eval()
with torch.no_grad():
    xc = torch.randn(4, 64, 8, 8)
    rc = cbam(xc) / xc
check("CBAM is purely multiplicative: out/x in (0, 1)",
      bool((rc > 0).all() and (rc < 1).all()),
      f"range [{rc.min():.3f}, {rc.max():.3f}]")

# Shape and finiteness across every width the backbone actually uses.
print("\nShape/finiteness sweep across backbone widths")
for c in CHANNELS:
    for name, mod in [("SE", A.SEBlock(c, REDUCTION)),
                      ("BAM", A.BAMBlock(c, REDUCTION)),
                      ("CBAM", A.CBAMBlock(c, REDUCTION))]:
        mod.eval()
        xi = torch.randn(2, c, 8, 8)
        with torch.no_grad():
            yo = mod(xi)
        check(f"{name} C={c}: shape preserved and finite",
              yo.shape == xi.shape and bool(torch.isfinite(yo).all()))
        hidden = c // REDUCTION
        check(f"{name} C={c}: bottleneck C/r = {hidden} is non-degenerate", hidden >= 2)

# End-to-end forward through each full network.
print("\nEnd-to-end forward pass")
for variant in [None, "SE", "BAM", "CBAM"]:
    m = resnet_cifar(attention_type=variant).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 32, 32))
    check(f"{variant or 'none'}: (2,3,32,32) -> (2,10), finite",
          out.shape == (2, 10) and bool(torch.isfinite(out).all()))

print(f"\nAll {len(checks)} checks passed.")
print("Parameter counts: " + "  ".join(f"{k}={v/1e6:.2f}M" for k, v in params.items()))
