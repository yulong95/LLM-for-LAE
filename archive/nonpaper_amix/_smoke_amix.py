"""Smoke test A-mix models forward/backward."""
import sys, os, torch
sys.path.insert(0, r'C:\Users\17859\Desktop\files\Grad_Project\LLM for LAE\Codes_v1')
from models.amix_models import TF_Amix, CNN_Amix
from models.amix_recovery import amix_to_V, hrate_from_H0
from utils import SMR_loss

device = torch.device('cuda:0')
N, K = 256, 10
H0 = torch.randn(4, K, N, 2, device=device)
cl = torch.randint(0, 2, (4, K, 1), device=device).float()

for name, m in [('TF', TF_Amix()), ('CNN', CNN_Amix())]:
    m = m.to(device)
    V, H_rate, clh, p, A = m(H0, cl)
    Rsum, single = SMR_loss(V, H_rate, 1e-2)
    loss = -Rsum.mean() + 5 * ((clh - cl.squeeze(-1)) ** 2).mean()
    loss.backward()
    print(name, 'V', tuple(V.shape), 'rate', Rsum.mean().item(), 'ok')

print('smoke ok')
