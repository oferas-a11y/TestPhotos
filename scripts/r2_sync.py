#!/usr/bin/env python3
"""
Sync local photos to Cloudflare R2 (S3-compatible) using boto3.

Uploads keys that mirror the original_path (full path) so the server can
serve them via IMAGE_CDN_BASE + original_path with IMAGE_PATH_MODE=fullpath.

Env vars (required):
- R2_ACCOUNT_ID: Cloudflare account ID (used for endpoint)
- R2_BUCKET: Bucket name
- R2_ACCESS_KEY_ID: Access key ID
- R2_SECRET_ACCESS_KEY: Secret key

Optional:
- SOURCE_DIR: defaults to 'photo_collections'
- DRY_RUN: set to '1' for a dry run summary

Example:
  pip install boto3
  export R2_ACCOUNT_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
  export R2_BUCKET=my-bucket
  export R2_ACCESS_KEY_ID=AKIA...
  export R2_SECRET_ACCESS_KEY=...
  python scripts/r2_sync.py

After upload, set on Railway:
  IMAGE_CDN_BASE=https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{R2_BUCKET}
  IMAGE_PATH_MODE=fullpath
  (then redeploy and test)
"""

import os
import sys
from pathlib import Path
from typing import Iterator, Tuple

def iter_images(root: Path) -> Iterator[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in exts:
            yield p

def human(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"

def main():
    try:
        import boto3
        from botocore.config import Config
    except Exception:
        print("Error: boto3 not installed. Run: pip install boto3")
        sys.exit(2)

    account_id = os.getenv('R2_ACCOUNT_ID')
    bucket = os.getenv('R2_BUCKET')
    access_key = os.getenv('R2_ACCESS_KEY_ID')
    secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
    if not all([account_id, bucket, access_key, secret_key]):
        print("Missing env. Required: R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(2)

    source = Path(os.getenv('SOURCE_DIR', 'photo_collections')).resolve()
    if not source.exists():
        print(f"Source not found: {source}")
        sys.exit(2)

    # Build endpoint
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    # Dry-run summary
    dry = os.getenv('DRY_RUN', '').strip() == '1'
    total = 0
    bytes_total = 0
    for p in iter_images(source):
        total += 1
        try:
            bytes_total += p.stat().st_size
        except Exception:
            pass
    print(f"Found {total} images under {source}")
    print(f"Estimated size: {human(bytes_total)}")
    if dry:
        print("DRY_RUN=1 set — not uploading.")
        print("To upload, unset DRY_RUN and run again.")
        sys.exit(0)

    # Upload
    uploaded = 0
    skipped = 0
    for p in iter_images(source):
        # Key mirrors the full path from project root
        # Convert absolute path to relative from project root
        try:
            rel = p.relative_to(Path.cwd())
        except Exception:
            # Fallback: relative to source
            rel = p.relative_to(source)
            rel = Path('photo_collections') / rel

        key = str(rel).replace('\\', '/')

        # Skip if exists with same size (best effort HEAD)
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            # If ContentLength matches, skip
            if head.get('ContentLength') == p.stat().st_size:
                skipped += 1
                continue
        except Exception:
            pass

        print(f"Uploading {key} ...")
        try:
            s3.upload_file(str(p), bucket, key, ExtraArgs={'ACL': 'public-read'})
            uploaded += 1
        except Exception as e:
            print(f"  ! Failed: {e}")

    print(f"Done. Uploaded: {uploaded}, skipped: {skipped}, total: {total}")
    print("Set on Railway:")
    print(f"  IMAGE_CDN_BASE={endpoint_url}/{bucket}")
    print("  IMAGE_PATH_MODE=fullpath")

if __name__ == '__main__':
    main()

