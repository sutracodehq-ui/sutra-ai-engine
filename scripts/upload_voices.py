import os
import boto3
from botocore.config import Config as BotoConfig
from pathlib import Path

# --- Constants ---
BUCKET = "digitamarketingaitraining"
ENDPOINT = "https://05432e110559ff6298b2fde4e6c68c01.r2.cloudflarestorage.com"
ACCESS_KEY = "7f36d9ab185d15ea60dad80db3a387cb"
SECRET_KEY = "89f25d3d72433515b7b05707719a7e5c719e8e686a86ce7e03732f4381fe7e78"

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    config=BotoConfig(signature_version="s3v4"),
    region_name="auto",
)

def upload_and_presign(file_path: Path):
    object_key = f"voice/demo/{file_path.name}"
    print(f"Uploading {file_path.name}...")
    
    with open(file_path, "rb") as f:
        s3.put_object(
            Bucket=BUCKET,
            Key=object_key,
            Body=f,
            ContentType="audio/mpeg"
        )
    
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": object_key},
        ExpiresIn=604800 # 7 days
    )
    return url

if __name__ == "__main__":
    files = list(Path("voice_demos").glob("greeting_*.mp3"))
    if not files:
        print("No greeting files found in voice_demos/")
        exit(1)
        
    print("\n--- R2 Playback URLs (Valid for 7 days) ---\n")
    for f in files:
        url = upload_and_presign(f)
        print(f"{f.name}: {url}\n")
