"""
Within-cancer survival engine.

Protocol v2 (unit = patient):
  * Training, like evaluation, is at PATIENT level: the risk score of a patient is the mean
    of the scores of its slides (differentiable), and the Breslow partial likelihood is
    computed over patients, so multi-slide patients are not counted several times and the
    training objective matches the patient-level C-index.
  * FL: FedAvg with client-local Breslow losses (risk sets = the client's own patients),
    one full-batch Adam step per round, aggregation weighted by the number of training
    patients; the global model is monitored on the pooled validation patients and the
    best-validation checkpoint is restored.
  * Centralised references: pooled Breslow likelihood and site-stratified Breslow likelihood
    (risk sets restricted to each Project_TSS site), full batch, same learning-rate grid and
    validation rule.
  * Ties in survival time share one risk set (Breslow); the loss is permutation invariant.

Legacy behaviour is reproduced by mode="fl_legacy_restore" / "central_legacy": slide-level
training with the legacy sequential loss; the FL server monitored a placeholder metric so
the round-1 model was restored; the centralised model was trained for 300 epochs without
validation.
"""
import time
from dataclasses import dataclass, asdict
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from .models import SurvivalMLP, cox_loss, stratified_cox_loss, cox_loss_legacy, n_params
from .metrics import harrell_c


@dataclass
class SurvConfig:
    fm: str = "UNI_v2"
    cancer: str = "BRCA"
    mode: str = "fl"            # fl | central | central_strat | fl_legacy_restore | central_legacy
    lr: float = 3e-4
    local_epochs: int = 1
    clients_per_round: Optional[int] = 20
    max_rounds: int = 1000
    patience: int = 100
    min_rounds: int = 100
    min_delta: float = 0.001
    unit: str = "patient"       # patient-level objective (v2); legacy modes are slide-level
    loss: str = "breslow"       # tie-correct Breslow (v2); legacy modes use the legacy loss
    seed: int = 0


def patient_c(risk, samples_idx, samples):
    by = defaultdict(list)
    for r, i in zip(risk, samples_idx):
        by[samples[i]["case_id"]].append(r)
    cases = sorted(by)
    rr = np.array([np.mean(by[c]) for c in cases])
    first = {}
    for i in samples_idx:
        first.setdefault(samples[i]["case_id"], i)
    t = np.array([samples[first[c]]["os_time"] for c in cases])
    e = np.array([samples[first[c]]["os_status"] for c in cases])
    return harrell_c(rr, t, e), cases, rr, t, e


class PatientGroup:
    """Slides of a set of patients, with the map slide -> local patient index."""

    def __init__(self, slide_idx, samples, device):
        cases = {}
        pat_of = []
        for i in slide_idx:
            c = samples[i]["case_id"]
            if c not in cases:
                cases[c] = len(cases)
            pat_of.append(cases[c])
        self.slides = torch.tensor(slide_idx, dtype=torch.long, device=device)
        self.pat_of = torch.tensor(pat_of, dtype=torch.long, device=device)
        self.n_pat = len(cases)
        self.counts = torch.zeros(self.n_pat, device=device).index_add_(0, self.pat_of, torch.ones(len(pat_of), device=device))
        first = {}
        for i in slide_idx:
            first.setdefault(samples[i]["case_id"], i)
        order = sorted(cases, key=cases.get)
        self.T = torch.tensor([samples[first[c]]["os_time"] for c in order], dtype=torch.float32, device=device)
        self.E = torch.tensor([samples[first[c]]["os_status"] for c in order], dtype=torch.float32, device=device)
        self.site_names = [samples[first[c]]["client_id"] for c in order]

    def risk(self, model, X):
        r = model(X[self.slides])
        return torch.zeros(self.n_pat, device=r.device).index_add_(0, self.pat_of, r) / self.counts


def run_survival(cfg: SurvConfig, X, samples, device="cuda"):
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    split = np.array([s["split"] for s in samples])
    T = torch.tensor([s["os_time"] for s in samples], dtype=torch.float32, device=device)
    E = torch.tensor([s["os_status"] for s in samples], dtype=torch.float32, device=device)
    tr = np.where(split == "train")[0]
    va = np.where(split == "val")[0]
    te = np.where(split == "test")[0]
    model = SurvivalMLP(X.shape[1]).to(device)
    legacy = cfg.mode in ("fl_legacy_restore", "central_legacy")

    @torch.no_grad()
    def risk_of(state, idx):
        model.load_state_dict(state)
        model.eval()
        return model(X[torch.tensor(idx, device=device)]).float().cpu().numpy()

    def val_c(state):
        return patient_c(risk_of(state, va), va, samples)[0]

    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    hist, best, best_r, best_state, no_imp = [], -1.0, 0, None, 0

    if cfg.mode in ("fl", "fl_legacy_restore"):
        clients = defaultdict(list)
        for i in tr:
            clients[samples[i]["client_id"]].append(i)
        cids = sorted(clients)
        groups = {c: PatientGroup(clients[c], samples, device) for c in cids}
        max_r = 100 if legacy else cfg.max_rounds
        for r in range(1, max_r + 1):
            k = cfg.clients_per_round
            sel = cids if (k is None or k >= len(cids)) else \
                [cids[j] for j in sorted(rng.choice(len(cids), k, replace=False))]
            acc = None
            tot = sum((len(clients[c]) if legacy else groups[c].n_pat) for c in sel)
            for c in sel:
                model.load_state_dict(state)
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
                for _ in range(cfg.local_epochs):                       # full batch
                    opt.zero_grad(set_to_none=True)
                    if legacy:
                        idx = torch.tensor(clients[c], device=device)
                        loss = cox_loss_legacy(model(X[idx]), T[idx], E[idx])
                    else:
                        g = groups[c]
                        loss = cox_loss(g.risk(model, X), g.T, g.E)
                    if not torch.isnan(loss):
                        loss.backward()
                        opt.step()
                w = (len(clients[c]) if legacy else groups[c].n_pat) / tot
                st = model.state_dict()
                if acc is None:
                    acc = {kk: v.detach().float() * w for kk, v in st.items()}
                else:
                    for kk, v in st.items():
                        acc[kk].add_(v.detach().float(), alpha=w)
            state = acc
            if legacy:
                if r == 1:
                    best_state, best_r = {k2: v.clone() for k2, v in state.items()}, 1
                continue
            v = val_c(state)
            hist.append(v)
            if v > best + cfg.min_delta:
                best, best_r, no_imp = v, r, 0
                best_state = {k2: t.clone() for k2, t in state.items()}
            else:
                no_imp += 1
            if no_imp >= cfg.patience and r >= cfg.min_rounds:
                break
    else:
        model.load_state_dict(state)
        lr = 1e-3 if legacy else cfg.lr
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        steps = 300 if legacy else cfg.max_rounds
        if legacy:
            idx = torch.tensor(tr, device=device)
        else:
            g = PatientGroup(list(tr), samples, device)
            site_ids = {c: i for i, c in enumerate(sorted(set(g.site_names)))}
            S = torch.tensor([site_ids[c] for c in g.site_names], device=device)
        for r in range(1, steps + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            if legacy:
                loss = cox_loss_legacy(model(X[idx]), T[idx], E[idx])
            else:
                risk = g.risk(model, X)
                loss = stratified_cox_loss(risk, g.T, g.E, S) if cfg.mode == "central_strat" else cox_loss(risk, g.T, g.E)
            if not torch.isnan(loss):
                loss.backward()
                opt.step()
            if legacy:
                continue
            cur = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            v = val_c(cur)
            hist.append(v)
            if v > best + cfg.min_delta:
                best, best_r, no_imp, best_state = v, r, 0, cur
            else:
                no_imp += 1
            if no_imp >= cfg.patience and r >= cfg.min_rounds:
                break
        if legacy:
            best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            best_r = steps

    risk_te = risk_of(best_state, te)
    c_pat, cases, rr, tt, ee = patient_c(risk_te, te, samples)
    c_slide = harrell_c(risk_te, T.cpu().numpy()[te], E.cpu().numpy()[te])
    rec = dict(config=asdict(cfg), best_round=best_r, best_val=best, val_history=hist,
               test_c_patient=c_pat, test_c_slide=c_slide, n_test_patients=len(cases),
               n_test_events=int(ee.sum()), n_params=n_params(model), seconds=time.time() - t0)
    pred = dict(cases=np.array(cases), risk=rr, time=tt, event=ee)
    return rec, pred
