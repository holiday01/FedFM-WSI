"""
Pairwise encoder differences with the same two-level bootstrap used for the
vs-best column of T_fm_ranking.csv.

The ranking table holds only differences against the best encoder of each
protocol; this script writes every pair, so that statements about any two
encoders can be checked against a table.

    python analysis/pairwise_encoders.py      ->  analysis/tables/T_fm_pairwise.csv
"""
import sys
from pathlib import Path
import itertools
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_cls as A   # noqa: E402


def main() -> None:
    df = A.load_all()
    out = []
    # Same two subsets main() uses: the fixed-rate Adam grid, and the SGD grid
    # with the learning rate selected on validation per encoder and algorithm.
    main_grid = df[df.grid == "main"]
    sgd_grid = df[df.grid == "sgd"]
    sel = A.select_by_val(sgd_grid, ["fm", "algorithm"], "lr") if len(sgd_grid) else sgd_grid
    protocols = {
        "adam": main_grid[main_grid.algorithm == "FedAvg"],
        "sgd": sel[sel.algorithm == "FedAvg"] if len(sel) else sel,
    }
    for pname, sub in protocols.items():
        fms = sorted(sub.fm.unique())
        pairs = list(itertools.combinations(fms, 2))
        if not pairs:
            continue
        rows, _ = A.diff_table([sub[sub.fm == a] for a, _ in pairs],
                               [sub[sub.fm == b] for _, b in pairs],
                               [f"{a}_vs_{b}" for a, b in pairs],
                               seed_key=f"pairwise:{pname}")
        for (a, b), r in zip(pairs, rows):
            r.update(protocol=pname, fm_a=a, fm_b=b,
                     excludes_zero=bool(r["lo"] > 0 or r["hi"] < 0))
            out.append(r)
    t = pd.DataFrame(out)
    p = A.OUT / "T_fm_pairwise.csv"
    t.to_csv(p, index=False)
    print(f"wrote {p} ({len(t)} pairs)")
    for pname, g in t.groupby("protocol"):
        print(f"\n--- {pname} ---")
        for _, r in g.iterrows():
            flag = "*" if r.excludes_zero else " "
            print(f" {flag} {r.fm_a:12s} - {r.fm_b:12s} {100*r['diff']:+6.1f} [{100*r['lo']:+6.1f},{100*r['hi']:+6.1f}]")


if __name__ == "__main__":
    main()
