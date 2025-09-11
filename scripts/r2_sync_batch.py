#!/usr/bin/env python3
"""
Batch uploader to Cloudflare R2 using S3 API, with resume support.

Reads R2 credentials from environment or .env (dotenv), and uploads a
slice of files so we can run multiple shorter uploads without hitting
session timeouts.

Env/args:
- R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (required)
- SOURCE_DIR (default: photo_collections)
- START (0-based index, default: 0)
- COUNT (number to upload, default: 200)

Usage examples:
  START=0 COUNT=200 python scripts/r2_sync_batch.py
  START=200 COUNT=200 python scripts/r2_sync_batch.py

Keys mirror full relative paths from repo root, e.g.:
  photo_collections/.../file.jpg

If a key exists with the same size, it is skipped.
"""

import os
import sys
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

def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in exts]
    files.sort(key=lambda p: str(p))
    return files

def main():
    account_id = os.getenv('R2_ACCOUNT_ID')
    bucket = os.getenv('R2_BUCKET')
    access_key = os.getenv('R2_ACCESS_KEY_ID')
    secret_key = os.getenv('R2_SECRET_ACCESS_KEY')
    if not all([account_id, bucket, access_key, secret_key]):
        print("Missing env. Required: R2_ACCOUNT_ID, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY")
        sys.exit(2)

    try:
        start = int(os.getenv('START', '0'))
        count = int(os.getenv('COUNT', '200'))
    except ValueError:
        print("Invalid START/COUNT values")
        sys.exit(2)

    source = Path(os.getenv('SOURCE_DIR', 'photo_collections'))
    if not source.exists():
        print(f"Source not found: {source}")
        sys.exit(2)

    files = list_images(source)
    total = len(files)
    end = min(start + count, total)
    if start >= total:
        print(f"START {start} beyond total {total}")
        sys.exit(0)

    print(f"Uploading files {start}..{end-1} of {total} from {source}")

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4')
    )

    uploaded = skipped = failed = 0
    for p in files[start:end]:
        try:
            rel = p.relative_to(Path.cwd())
        except Exception:
            rel = p.relative_to(source)
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

    print(f"Done. Uploaded={uploaded}, Skipped={skipped}, Failed={failed}, Range={start}-{end-1}")

if __name__ == '__main__':
    main()

