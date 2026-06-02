"""Upload label files to Google Drive via MCP-compatible requests using the claude.ai session.

Since direct API auth is unavailable, we zip the labels and print base64 for manual MCP upload.
"""
import base64, zipfile, io
from pathlib import Path

DATA_BASE = Path("data/pseudo_labels/all_4videos")

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
    for split in ("train", "val"):
        for p in sorted((DATA_BASE / "labels" / split).glob("*.txt")):
            zf.write(p, f"labels/{split}/{p.name}")

buf.seek(0)
data = buf.read()
print(f"Zip size: {len(data)/1024:.1f} KB")
b64 = base64.b64encode(data).decode()
out = Path("/tmp/labels_b64.txt")
out.write_text(b64)
print(f"Base64 written to {out}  ({len(b64)} chars)")
