import os
from pathlib import Path
import boto3
from botocore.config import Config

ACCOUNT_ID='0c6e85d3cb9e92105b01f4108662ac71'
BUCKET='ofers'
ACCESS_KEY='eec869358450beebf49ec166bb89608f'
SECRET_KEY='4cce3637ca69984b30237542ea38aed0c907eb531d81eab4797edb824d414488'
ENDPOINT=f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com'

s3=boto3.client('s3',endpoint_url=ENDPOINT,aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY,config=Config(signature_version='s3v4'))

root=Path('photo_collections')
imgs=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}]
imgs=imgs[:5]
print('Uploading',len(imgs),'files...')
for p in imgs:
    key=str(p).replace('\\\\','/')
    try:
        s3.upload_file(str(p), BUCKET, key)
        print('ok', key)
    except Exception as e:
        print('failed', key, e)

if imgs:
    print('Example URL:')
    print(f"{ENDPOINT}/{BUCKET}/"+str(imgs[0]).replace('\\\\','/'))
