"""Upload dataset + notebook to Google Drive using application default credentials."""
import sys
from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import google.auth

FOLDER_IDS = {
    "images/train": "1Ccv0Lmtt2wEHSfNNIHVTlPkWg0REKdC4",
    "images/val":   "1AQAKSpegxySedpDzMkdDyiaUGDHMYRrc",
    "labels/train": "14ynt3FKT9FFyC25aDk829u06VWJkmWkB",
    "labels/val":   "1CjJYkMMF4dpZenR_u03e03rhW2HFP3B7",
    "root":         "1_n2AlNxW_GGCPMOk8bvFd3sxLCln_WpF",
}

DATA_BASE = Path("data/pseudo_labels/all_4videos")
NOTEBOOK  = Path("notebooks/train_yolo26l_colab.ipynb")

def upload_file(service, path: Path, parent_id: str, mime: str):
    meta = {"name": path.name, "parents": [parent_id]}
    media = MediaFileUpload(str(path), mimetype=mime, resumable=True)
    f = service.files().create(body=meta, media_body=media, fields="id,name").execute()
    return f

def main():
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/drive.file"])
    service = build("drive", "v3", credentials=creds)

    # Upload notebook
    print(f"Uploading notebook...")
    upload_file(service, NOTEBOOK, FOLDER_IDS["root"],
                "application/octet-stream")
    print(f"  ✓ {NOTEBOOK.name}")

    # Upload labels (small, plain text)
    for split in ("train", "val"):
        lbl_dir = DATA_BASE / "labels" / split
        files = sorted(lbl_dir.glob("*.txt"))
        folder_id = FOLDER_IDS[f"labels/{split}"]
        print(f"Uploading labels/{split}: {len(files)} files...")
        for i, f in enumerate(files, 1):
            upload_file(service, f, folder_id, "text/plain")
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

    # Upload images (large, JPEG)
    for split in ("train", "val"):
        img_dir = DATA_BASE / "images" / split
        files = sorted(img_dir.glob("*.jpg"))
        folder_id = FOLDER_IDS[f"images/{split}"]
        print(f"Uploading images/{split}: {len(files)} files...")
        for i, f in enumerate(files, 1):
            upload_file(service, f, folder_id, "image/jpeg")
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
