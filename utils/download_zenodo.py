#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════════════
# DELPHI — Zenodo Download Script
# Institute for Protein Innovation (IPI)
#
# Downloads all files from a Zenodo record.
# Supports both published records and unpublished deposits (preview mode).
#
# Usage:
#   # Published record (no token needed)
#   python utils/download_zenodo.py
#
#   # Unpublished/preview deposit (token required)
#   python utils/download_zenodo.py --token YOUR_TOKEN
#
#   # Dry-run: list files without downloading
#   python utils/download_zenodo.py --token YOUR_TOKEN --dry-run
#
#   # Download only matching files
#   python utils/download_zenodo.py --token YOUR_TOKEN --filter psr_filter
#
# Generate token at:
#   https://zenodo.org/account/settings/applications/tokens/new/
#   Scopes required: deposit:read
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

RECORD_ID  = "20752840"
ZENODO_DOI = "10.5281/zenodo.20752840"
ZENODO_URL = f"https://zenodo.org/records/{RECORD_ID}"
_ROOT      = Path(__file__).resolve().parent.parent
OUT_DIR    = _ROOT / "pretrained_202605"   # default: save IPI models here

# API endpoints
API_DEPOSIT = "https://zenodo.org/api/deposit/depositions/{record_id}"
API_FILES   = "https://zenodo.org/api/deposit/depositions/{record_id}/files"
API_RECORD  = "https://zenodo.org/api/records/{record_id}"


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
    else:
        mb = (block_num * block_size) / 1e6
        print(f"\r  {mb:.1f} MB downloaded...", end="", flush=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def _get_json(url: str, token: str = None) -> dict:
    """GET a URL with optional Bearer token, return parsed JSON."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_files(record_id: str, token: str = None) -> list:
    """
    Fetch file list. Tries deposit API first (for unpublished/preview),
    falls back to records API (for published).
    """
    # ── Try deposit/depositions API (works for unpublished) ───────────────
    if token:
        url = API_FILES.format(record_id=record_id)
        try:
            print(f"  Trying deposit API (for preview records)...")
            data = _get_json(url, token=token)
            # Deposit files API returns a list directly
            if isinstance(data, list) and data:
                print(f"  Found {len(data)} files via deposit API")
                # Convert to standard format
                files = []
                for f in data:
                    files.append({
                        "key":   f.get("filename", f.get("key", "")),
                        "size":  f.get("filesize", f.get("size", 0)),
                        "links": {
                            "self": f.get("links", {}).get("download", "")
                        }
                    })
                return files
        except urllib.error.HTTPError as e:
            print(f"  Deposit API returned {e.code} — trying records API...")

    # ── Try records API (works for published) ─────────────────────────────
    url = API_RECORD.format(record_id=record_id)
    if token:
        url += f"?access_token={token}"
    try:
        print(f"  Trying records API...")
        data  = _get_json(url, token=token)
        title = data.get("metadata", {}).get("title", "Unknown")
        doi   = data.get("doi", "N/A")
        print(f"  Title : {title}")
        print(f"  DOI   : {doi}")

        # New Zenodo API: files under data["files"]["entries"]
        entries = data.get("files", {})
        if isinstance(entries, dict):
            file_list = list(entries.get("entries", {}).values())
        elif isinstance(entries, list):
            file_list = entries
        else:
            file_list = []
        return file_list

    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"\n  ERROR 403: Access denied to record {record_id}.")
            print(f"  This record is in preview (unpublished).")
            print(f"  Generate a token at:")
            print(f"    https://zenodo.org/account/settings/applications/tokens/new/")
            print(f"  Scopes needed: deposit:read")
            print(f"  Then run: python utils/download_zenodo.py --token YOUR_TOKEN")
            sys.exit(1)
        if e.code == 404:
            print(f"\n  ERROR 404: Record {record_id} not found.")
            sys.exit(1)
        raise


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

    # Append token to URL for authenticated download
    if token:
        sep  = "&" if "?" in url else "?"
        url += f"{sep}access_token={token}"

    dest = out_dir / fname

    if dest.exists():
        local_size = dest.stat().st_size
        if size and abs(local_size - size) < 1024:
            print(f"  SKIP  {fname}  ({size_mb:.1f} MB) — already exists")
            return True
        else:
            print(f"  RE-DL {fname}  "
                  f"(local={local_size/1e6:.1f} MB, expected={size_mb:.1f} MB)")

    if dry_run:
        print(f"  [DRY]  {fname:<60} {size_mb:>8.1f} MB")
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
        description="Download all files from a Zenodo record (published or preview)")
    ap.add_argument("--record",  default=RECORD_ID,
                    help=f"Zenodo record ID (default: {RECORD_ID})")
    ap.add_argument("--token",   default=None,
                    help="Personal access token (required for preview/unpublished records). "
                         "Generate at: https://zenodo.org/account/settings/applications/tokens/new/")
    ap.add_argument("--outdir",  default=str(OUT_DIR),
                    help="Output directory (default: pretrained_202605/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files without downloading")
    ap.add_argument("--filter",  default=None,
                    help="Only download files whose name contains this string "
                         "(e.g. --filter psr_filter  or  --filter .pt)")
    ap.add_argument("--include-embeddings", action="store_true",
                    help="Also download DS1 embedding files (DS1*.emb.csv). "
                         "These are large and not needed for prediction. "
                         "Excluded by default.")
    args = ap.parse_args()

    out_dir = Path(args.outdir)

    print()
    print("══════════════════════════════════════════════════════════════════")
    print("  DELPHI — Zenodo Download")
    print(f"  Record  : {ZENODO_URL}")
    print(f"  DOI     : {ZENODO_DOI}")
    print(f"  Output  : {out_dir}")
    print(f"  Token   : {'provided' if args.token else 'none (public only)'}")
    if args.dry_run:
        print("  Mode    : DRY-RUN")
    emb_status = "included" if args.include_embeddings else "excluded (use --include-embeddings)"
    print(f"  Embeddings: DS1*.emb.csv {emb_status}")
    if args.filter:
        print(f"  Filter  : '{args.filter}'")
    print("══════════════════════════════════════════════════════════════════")
    print()

    # Get file list
    files = get_files(args.record, token=args.token)

    if not files:
        print("  No files found.")
        sys.exit(0)

    # Apply filter
    if args.filter:
        files = [f for f in files
                 if args.filter in (f.get("key") or f.get("filename", ""))]
        print(f"  After filter '{args.filter}': {len(files)} files")

    # Exclude DS1 embedding CSVs by default (large, not needed for prediction)
    if not args.include_embeddings:
        before = len(files)
        files  = [f for f in files
                  if ".emb.csv" not in (f.get("key") or f.get("filename", ""))]
        skipped = before - len(files)
        if skipped:
            print(f"  Skipped {skipped} DS1 embedding file(s) "
                  f"(use --include-embeddings to download them)")

    if not files:
        print("  No files match the filter.")
        sys.exit(0)

    # Print file list
    total_size = sum(f.get("size", 0) or f.get("filesize", 0) for f in files)
    print(f"\n  Files ({len(files)} total, {total_size/1e9:.2f} GB):\n")
    for f in files:
        fname = f.get("key") or f.get("filename", "unknown")
        size  = f.get("size", 0) or f.get("filesize", 0)
        print(f"    {fname:<70} {size/1e6:>8.1f} MB")
    print()

    if args.dry_run:
        print("  Dry-run complete. Remove --dry-run to download.")
        return

    # Download
    out_dir.mkdir(parents=True, exist_ok=True)
    print("── Downloading ─────────────────────────────────────────────────")

    n_ok = n_fail = 0
    for f in files:
        ok = download_file(f, out_dir, token=args.token, dry_run=False)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

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
