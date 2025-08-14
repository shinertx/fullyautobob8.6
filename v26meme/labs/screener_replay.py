# Simple PIT screener snapshot browser/export (optional utility).
import json, sys
from pathlib import Path

def list_snapshots(dir_path="data/screener_snapshots"):
    p = Path(dir_path)
    for f in sorted(p.glob("*.json")):
        print(f)

def export_latest(dir_path="data/screener_snapshots", out="latest_panel.json"):
    p = Path(dir_path)
    files = sorted(p.glob("*.json"))
    if not files:
        print("No snapshots.")
        return
    data = json.loads(files[-1].read_text())
    Path(out).write_text(json.dumps(data))
    print(f"Exported to {out}")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "list"
    if cmd == "list": list_snapshots()
    else: export_latest()
