#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Zenodo Download Script
# Institute for Protein Innovation (IPI)
#
# Two modes:
#   Default          : download released DELPHI model artifacts from Zenodo
#   --embeddings     : download DS1_embedding.tar.gz → extract to data/
#
# Usage:
#   python utils/download_zenodo.py               # all pretrained models
#   python utils/download_zenodo.py --embeddings  # DS1 embeddings → data/
#   python utils/download_zenodo.py --dry-run     # preview without downloading
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

RECORD_ID  = "20785877"
ZENODO_DOI = "10.5281/zenodo.20785877"
ZENODO_URL = f"https://zenodo.org/records/{RECORD_ID}"
DS1_EMB_URL = "https://zenodo.org/records/20785877/files/DS1_embedding.tar.gz?download=1"
_ROOT      = Path(__file__).resolve().parent.parent
OUT_DIR    = _ROOT / "pretrained_202605"

API_RECORD  = "https://zenodo.org/api/records/{record_id}"
API_DEPOSIT = "https://zenodo.org/api/deposit/depositions/{record_id}/files"


# ── Progress bar ──────────────────────────────────────────────────────────────
def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100 / total_size)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        mb  = downloaded / 1e6
        tot = total_size / 1e6
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:.1f}/{tot:.1f} MB",
              end="", flush=True)


# ── Zenodo API — with pagination ──────────────────────────────────────────────
def _get_json(url: str, token: str = None) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_all_files(record_id: str, token: str = None) -> list:
    """
    Fetch ALL files from a Zenodo record, handling pagination.
    Tries deposit API first (for unpublished), then records API.
    """
    # ── Try deposit API (unpublished/preview) ─────────────────────────────
    if token:
        url = API_DEPOSIT.format(record_id=record_id)
        try:
            print("  Trying deposit API (preview mode)...")
            data = _get_json(url, token=token)
            if isinstance(data, list) and data:
                files = [{"key":  f.get("filename", f.get("key", "")),
                          "size": f.get("filesize", f.get("size", 0)),
                          "links": {"self": f.get("links", {}).get("download", "")}}
                         for f in data]
                print(f"  Found {len(files)} files via deposit API")
                return files
        except urllib.error.HTTPError:
            pass

    # ── Records API with pagination ───────────────────────────────────────
    print("  Fetching file list from Zenodo API (with pagination)...")
    all_files = []
    page = 1
    page_size = 100   # Zenodo max per page

    while True:
        url = f"{API_RECORD.format(record_id=record_id)}?page={page}&size={page_size}"
        if token:
            url += f"&access_token={token}"

        try:
            data = _get_json(url, token=token)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print(f"\n  ERROR 403: Access denied — use --token for preview records")
                print(f"  Generate token: https://zenodo.org/account/settings/applications/")
                sys.exit(1)
            raise

        title = data.get("metadata", {}).get("title", "Unknown")
        doi   = data.get("doi", "N/A")

        if page == 1:
            print(f"  Title : {title}")
            print(f"  DOI   : {doi}")

        # Parse files from response — handle both old and new Zenodo API formats
        entries = data.get("files", [])
        if isinstance(entries, dict):
            # New InvenioRDM format: {"entries": {...}}
            entries = list(entries.get("entries", {}).values())

        if not entries:
            break

        all_files.extend(entries)

        # Check if more pages exist
        if len(entries) < page_size:
            break
        page += 1

    print(f"  Total files found: {len(all_files)}")
    return all_files


# ── Download ──────────────────────────────────────────────────────────────────
def download_file(file_info: dict, out_dir: Path,
                  token: str = None, dry_run: bool = False) -> bool:
    fname    = file_info.get("key") or file_info.get("filename", "unknown")
    size     = file_info.get("size", 0) or file_info.get("filesize", 0)
    links    = file_info.get("links", {})
    url      = (links.get("self") or links.get("download") or
                links.get("content") or file_info.get("url", ""))
    size_mb  = size / 1e6 if size else 0

    if not url:
        print(f"  WARNING: No URL for {fname} — skipping")
        return False

    if token:
        sep  = "&" if "?" in url else "?"
        url += f"{sep}access_token={token}"

    dest = out_dir / fname

    if dest.exists():
        local_size = dest.stat().st_size
        if size and abs(local_size - size) < 1024:
            print(f"  SKIP  {fname}  ({size_mb:.1f} MB) — already exists")
            return True
        print(f"  RE-DL {fname}  (local={local_size/1e6:.1f} MB, expected={size_mb:.1f} MB)")

    if dry_run:
        print(f"  [DRY]  {fname:<65} {size_mb:>8.1f} MB")
        return True

    print(f"\n  Downloading: {fname}  ({size_mb:.1f} MB)")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        elapsed = time.time() - t0
        speed   = size / elapsed / 1e6 if elapsed > 0 and size else 0
        print(f"  OK  {fname}  ({elapsed:.0f}s  {speed:.1f} MB/s)")
        return True
    except Exception as e:
        print(f"\n  ERROR: {fname}: {e}")
        if dest.exists():
            dest.unlink()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Download DELPHI pretrained models from Zenodo")
    ap.add_argument("--embeddings", action="store_true",
                    help="Download DS1_embedding.tar.gz and extract all "
                         "DS1 embedding CSVs + sequence xlsx to data/. "
                         "Default mode downloads released DELPHI model artifacts from Zenodo.")
    ap.add_argument("--outdir",  default=str(OUT_DIR),
                    help=f"Output directory (default: pretrained_202605/)")
    ap.add_argument("--token",   default=None,
                    help="Access token for preview/unpublished records")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files without downloading")
    args = ap.parse_args()

    out_dir = Path(args.outdir)

    mode = ("DS1 embeddings → data/  (DS1_embedding.tar.gz)"
            if args.embeddings else
            "released DELPHI model artifacts")

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DELPHI — Zenodo Download")
    print(f"  Record  : {ZENODO_URL}")
    print(f"  DOI     : {ZENODO_DOI}")
    print(f"  Mode    : {mode}")
    print(f"  Output  : {out_dir}")
    if args.dry_run:
        print("  Dry-run : yes (no files downloaded)")
    print("══════════════════════════════════════════════════════════════════")
    print()

    # Get all files with pagination
    files = get_all_files(RECORD_ID, token=args.token)

    if not files:
        print("  No files found in this record.")
        sys.exit(0)

    SKIP_DEFAULT = {"DS1.xlsx", "DS1_embedding.tar.gz"}
    IS_DS1_DATA = lambda f: (
        (f.get("key") or f.get("filename", "")) in SKIP_DEFAULT or
        ".emb.csv" in (f.get("key") or f.get("filename", ""))
    )

    if args.embeddings:
        # ── Download DS1_embedding.tar.gz → extract to data/ ─────────────
        data_dir = _ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        tar_path = data_dir / "DS1_embedding.tar.gz"

        print(f"\n── Downloading DS1 embeddings ──────────────────────────────")
        print(f"  URL    : {DS1_EMB_URL}")
        print(f"  Target : {data_dir}/")

        print(f"\n  Downloading DS1_embedding.tar.gz...")
        t0 = time.time()
        urllib.request.urlretrieve(DS1_EMB_URL, tar_path, reporthook=_progress)
        print()
        elapsed = time.time() - t0
        size_mb = tar_path.stat().st_size / 1e6
        print(f"  OK  DS1_embedding.tar.gz  ({size_mb:.1f} MB  {elapsed:.0f}s)")

        print(f"\n  Extracting to {data_dir}/...")
        import tarfile
        with tarfile.open(tar_path) as tar:
            members = tar.getmembers()
            print(f"  Files in archive: {len(members)}")
            for m in members:
                print(f"    {m.name}  ({m.size/1e6:.1f} MB)")
            tar.extractall(data_dir)

        # Move any nested files up to data/ root
        for sub in data_dir.iterdir():
            if sub.is_dir():
                for f in sub.iterdir():
                    dest = data_dir / f.name
                    if not dest.exists():
                        f.rename(dest)
                        print(f"  Moved: {f.name}")
                sub.rmdir() if not list(sub.iterdir()) else None

        tar_path.unlink()   # remove archive after extraction
        print(f"\n  Extracted files in data/:")
        for f in sorted(data_dir.iterdir()):
            print(f"    {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

        print()
        print("══════════════════════════════════════════════════════════════════")
        print("  Done.  DS1 embeddings extracted → data/")
        print("══════════════════════════════════════════════════════════════════")
        print()
        return

    else:
        # Default: released DELPHI model artifacts.
        # The live Zenodo record may contain more files than the registry; do
        # not hard-code a model count here.
        emb_count = sum(1 for f in files if IS_DS1_DATA(f))
        files     = [f for f in files if not IS_DS1_DATA(f)]
        print(f" Selected {len(files)} released DELPHI model artifact file(s).")
        if emb_count:
            print(" DS1 public data and embeddings are available with --embeddings.")

    # ── Print file list ───────────────────────────────────────────────────
    total_size = sum(f.get("size", 0) or f.get("filesize", 0) for f in files)
    print(f"\n  Files to download: {len(files)}  ({total_size/1e9:.2f} GB)\n")
    for f in sorted(files, key=lambda x: x.get("key") or ""):
        fname = f.get("key") or f.get("filename", "unknown")
        size  = f.get("size", 0) or f.get("filesize", 0)
        print(f"    {fname:<70} {size/1e6:>8.1f} MB")
    print()

    if args.dry_run:
        print("  Dry-run complete — remove --dry-run to download.")
        return

    # ── Download ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    print("── Downloading ─────────────────────────────────────────────────")

    n_ok = n_fail = 0
    for f in sorted(files, key=lambda x: x.get("key") or ""):
        ok = download_file(f, out_dir, token=args.token, dry_run=False)
        if ok: n_ok += 1
        else:  n_fail += 1

    print()
    print("══════════════════════════════════════════════════════════════════")
    print(f"  Done.  OK={n_ok}  Failed={n_fail}")
    print(f"  Output : {out_dir}")
    if n_fail:
        print(f"  {n_fail} failed — re-run to retry")
    print("══════════════════════════════════════════════════════════════════")
    print()


if __name__ == "__main__":
    main()
