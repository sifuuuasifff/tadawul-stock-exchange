import json
from config.settings import MEMORY_DIR
with open(MEMORY_DIR / "all_current_states.json") as f:
    states = json.load(f)["stocks"]
j = states.get("4015", {})
qr = j.get("quarterly_records", [])
print(f"Jamjoom quarterly records embedded in JSON: {len(qr)}")
for r in qr[-5:]:
    print(f"  {r}")
print(f"\nTTM summary: {j.get('ttm_summary', {})}")
