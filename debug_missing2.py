import sys; sys.path.insert(0, ".")
import json
from config.settings import MEMORY_DIR

with open(MEMORY_DIR / "all_current_states.json") as f:
    states = json.load(f)["stocks"]

s = states.get("4015", {})
ttm = s.get("ttm_summary", {})
print("TTM summary keys:", list(ttm.keys()))
print("TTM summary:", json.dumps(ttm, indent=2))
print("\nquarterly_records sample (last 2):")
for r in s.get("quarterly_records", [])[-2:]:
    print(r)
