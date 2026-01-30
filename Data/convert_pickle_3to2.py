import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def p(*parts) -> Path:
    return REPO_ROOT.joinpath(*parts)


# Input: adjust to the real file you want to convert (relative to repo root)
in_path = p("Data", "orgs", "processed_data", "orgs_data_ix", "full_ix_org_inv.dict")

# Output:
out_path = p("Data", "orgs", "processed_data", "orgs_data_ix", "items_for_pred.pkl")

out_path.parent.mkdir(parents=True, exist_ok=True)

with open(in_path, "rb") as f:
    obj = pickle.load(f)

with open(out_path, "wb") as f:
    pickle.dump(obj, f, protocol=2)

print(f"Saved: {out_path}")
