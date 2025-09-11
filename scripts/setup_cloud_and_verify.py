#!/usr/bin/env python3
"""
End-to-end setup:
1) Upload ALL local photos to Cloudflare R2 (S3 API), using full original_path keys
2) Verify one public file URL
3) Print Railway env commands and verify the live server returns a real photo

Reads configuration from .env (dotenv):
  R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
  IMAGE_CDN_BASE (optional for verify), IMAGE_PATH_MODE (should be 'fullpath')
  BASE_URL (your deployed API base, optional for test)

This script is resumable: it skips keys that already exist with the same size.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import boto3
    from botocore.config import Config
except Exception:
    print("Error: boto3 not installed. Run: pip install boto3")
    sys.exit(2)

from urllib.request import urlopen, Request


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in exts]
    files.sort(key=lambda p: str(p))
    return files


def ensure_env(*keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        print("Missing env keys:", ", ".join(missing))
        sys.exit(2)


def upload_all(account_id: str, bucket: str, access_key: str, secret_key: str, source_dir: Path) -> int:
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )
    files = list_images(source_dir)
    total = len(files)
    print(f"Found {total} images under {source_dir}")

    uploaded = skipped = failed = 0
    for p in files:
        try:
            rel = p.relative_to(Path.cwd())
        except Exception:
            rel = p.relative_to(source_dir)
            rel = Path('photo_collections') / rel
        key = str(rel).replace('\\\\', '/')

        # Skip if exists with same size
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if head.get('ContentLength') == p.stat().st_size:
                skipped += 1
                continue
        except Exception:
            pass

        try:
            s3.upload_file(str(p), bucket, key)
            uploaded += 1
        except Exception as e:
            failed += 1
            print(f"Failed {key}: {e}")

    print(f"Upload complete. Uploaded={uploaded}, Skipped={skipped}, Failed={failed}, Total={total}")
    return total - failed


def try_public_fetch(base: str, sample_path: str) -> bool:
    url = base.rstrip('/') + '/' + sample_path.replace(' ', '%20')
    try:
        req = Request(url, headers={'User-Agent': 'curl/8'})
        with urlopen(req, timeout=30) as r:
            data = r.read()
            print(f"Public fetch OK: {url} -> {len(data)} bytes, {r.headers.get('Content-Type')}")
            return True
    except Exception as e:
        print(f"Public fetch failed: {url} -> {e}")
        return False


def main():
    ensure_env('R2_ACCOUNT_ID', 'R2_BUCKET', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY')
    account = os.getenv('R2_ACCOUNT_ID')
    bucket = os.getenv('R2_BUCKET')
    access = os.getenv('R2_ACCESS_KEY_ID')
    secret = os.getenv('R2_SECRET_ACCESS_KEY')
    source = Path(os.getenv('SOURCE_DIR', 'photo_collections'))
    if not source.exists():
        print(f"Source not found: {source}")
        sys.exit(2)

    # 1) Upload all files (resumable)
    ok = upload_all(account, bucket, access, secret, source)

    # Pick a sample path for verification
    files = list_images(source)
    sample = files[len(files)//2] if files else None
    if not sample:
        print('No files found to verify')
        sys.exit(1)
    try:
        rel = sample.relative_to(Path.cwd())
    except Exception:
        rel = sample.relative_to(source)
        rel = Path('photo_collections') / rel
    sample_key = str(rel).replace('\\\\', '/')

    # 2) Verify public access (requires bucket to be public)
    cdn_base = os.getenv('IMAGE_CDN_BASE') or f"https://{account}.r2.cloudflarestorage.com/{bucket}"
    print("\nVerifying public access (bucket must be public):")
    try_public_fetch(cdn_base, sample_key)

    # 3) Print Railway env setup and test command
    print("\nSet these in Railway → Variables and redeploy:")
    print(f"  IMAGE_CDN_BASE={cdn_base}")
    print("  IMAGE_PATH_MODE=fullpath")
    base_url = os.getenv('BASE_URL')
    if base_url:
        print("\nAfter deploy, verify server returns a real image:")
        print(f"  BASE_URL={base_url} python scripts/test_server_photo.py")
    else:
        print("\nTip: add BASE_URL=https://<your-railway> to .env to enable the test script.")

    print("\nDone. If placeholders persist, ensure the bucket is public and keys match metadata original_path.")


if __name__ == '__main__':
    main()

