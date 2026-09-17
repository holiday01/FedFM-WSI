"""
Centralised baselines.

  protocol="tuned"  [v2]: same head, class-weighted CE, Adam or SGD at the given lr, mini-batch 32,
                    validation accuracy after every epoch, early stopping with the same rule as FL
                    (patience `patience` epochs, at least `min_epochs`, at most `epochs`),
                    best-validation checkpoint restored.
  protocol="legacy"     : legacy baseline (Adam lr=1e-3, unweighted CE, 300 epochs,
                    last epoch evaluated, no validation).
"""
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from .fl import make_model, transform_features, NUM_CLASSES
from .metrics import classification_metrics
from .models import n_params


@dataclass
class CentralConfig:
    fm: str = "UNI_v2"
    protocol: str = "tuned"
    optimizer: str = "adam"              # adam | sgd
    momentum: float = 0.0
    lr: float = 3e-4
    epochs: int = 300
    patience: int = 50
    min_epochs: int = 100
    batch_size: int = 32
    arch: str = "mlp"
    hidden_dims: tuple = (512, 256)
    dropout: float = 0.3
    class_weighting: bool = True
    feature_transform: str = "raw"
    rp_dim: int = 6144
    seed: int = 0
    tag: str = ""


def run_central(cfg: CentralConfig, X, y, samples, device="cuda"):
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    split = np.array([s["split"] for s in samples])
    train_mask = torch.tensor(split == "train", device=device)
    Xt = transform_features(X, train_mask, cfg, device)
    tr = torch.tensor(np.where(split == "train")[0], device=device)
    va = torch.tensor(np.where(split == "val")[0], device=device)
    te = torch.tensor(np.where(split == "test")[0], device=device)
    legacy = cfg.protocol == "legacy"
    model = make_model(cfg, Xt.shape[1], batch_norm=False).to(device)
    lr = 1e-3 if legacy else cfg.lr
    if legacy or cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum)
    if cfg.class_weighting and not legacy:
        cnt = np.bincount(y[tr].cpu().numpy(), minlength=NUM_CLASSES)
        w = [len(tr) / (NUM_CLASSES * c) if c > 0 else 1.0 for c in cnt]
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def acc(idx):
        model.eval()
        out = torch.cat([model(Xt[idx[s:s + 2048]]) for s in range(0, idx.numel(), 2048)])
        model.train()
        return out

    best, best_ep, best_state, hist, no_imp = -1.0, 0, None, [], 0
    n, bs = tr.numel(), cfg.batch_size
    model.train()
    for ep in range(1, cfg.epochs + 1):
        perm = tr[torch.randperm(n, device=device)]
        for b in range(n // bs):                       # DataLoader(drop_last=True) as in the legacy code
            idx = perm[b * bs:(b + 1) * bs]
            opt.zero_grad(set_to_none=True)
            crit(model(Xt[idx]), y[idx]).backward()
            opt.step()
        if not legacy:
            v = float((acc(va).argmax(1) == y[va]).float().mean())
            hist.append(v)
            if v > best + 1e-3:
                best, best_ep, no_imp = v, ep, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                no_imp += 1
            if no_imp >= cfg.patience and ep >= cfg.min_epochs:                 # same rule as FL
                break
    if not legacy:
        model.load_state_dict(best_state)
    lg = acc(te)
    probs = torch.softmax(lg.float(), 1).cpu().numpy()
    te_np = te.cpu().numpy()
    met = classification_metrics(y[te].cpu().numpy(), probs.argmax(1),
                                 np.array([samples[i]["client_id"] for i in te_np]))
    return dict(config=asdict(cfg), best_epoch=best_ep, best_val=best, val_history=hist, epochs_run=len(hist) if not legacy else cfg.epochs,
                total_steps=(len(hist) if not legacy else cfg.epochs) * (n // bs),
                test=met, n_params=n_params(model), seconds=time.time() - t0), \
        dict(test_idx=te_np, probs=probs.astype(np.float16)), model.state_dict()
