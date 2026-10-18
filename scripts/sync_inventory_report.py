"""
sync_inventory_report.py
========================
Quick utility: copies the latest inventory report JSON from
  output/inventory_report/inventory_report.json
to
  dashboard/public/inventory_report.json

Run this after executing demo_inventory_report.py to refresh the dashboard.
Usage:
  python scripts/sync_inventory_report.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "output" / "inventory_report" / "inventory_report.json"
DEST = ROOT / "dashboard" / "public" / "inventory_report.json"

def main():
    if not SRC.exists():
        print(f"[ERROR] Source not found: {SRC}")
        print("  Run scripts/demo_inventory_report.py first.")
        return

    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DEST)
    size_kb = DEST.stat().st_size / 1024
    print(f"[OK] Synced {SRC.name} -> {DEST} ({size_kb:.1f} KB)")
    print("     Reload the dashboard to see updated data.")

if __name__ == "__main__":
    main()
