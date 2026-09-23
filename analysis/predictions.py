"""Single reader for the archived per-slide predictions of a classification run (<key>_pred.npz).

Source rule (2026-09-22): the integer class assigned at evaluation (`pred`) is authoritative when present; runs archived
before 2026-09-22 store float16 probabilities only, and their class is the argmax of those probabilities.  Every analysis
branch (tables, bootstrap, per-client accuracy, checkpoint diagnostic, verification) must read predictions through here.
"""
import numpy as np


def load_predictions(npz_path):
    p = np.load(npz_path)
    idx = p["test_idx"]
    pred = p["pred"].astype(int) if "pred" in p.files else p["probs"].astype(np.float32).argmax(1)
    return idx, pred


def archived_predictions(json_path):
    """(test_idx, predicted class) of the run whose JSON is `json_path`."""
    return load_predictions(str(json_path).replace(".json", "_pred.npz"))
