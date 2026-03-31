import sys
from pathlib import Path

# Ensure project root is in sys.path so app.config is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import boto3
from botocore.config import Config as BotoConfig
from app.config import get_settings

settings = get_settings()

s3 = boto3.client(
    "s3",
    endpoint_url=settings.r2_endpoint,
    aws_access_key_id=settings.r2_access_key,
    aws_secret_access_key=settings.r2_secret_key,
    config=BotoConfig(signature_version="s3v4"),
    region_name="auto",
)

def upload_and_presign(file_path: Path):
    object_key = f"voice/demo/{file_path.name}"
    print(f"Uploading {file_path.name}...")
    
    with open(file_path, "rb") as f:
        s3.put_object(
            Bucket=settings.r2_bucket,
            Key=object_key,
            Body=f,
            ContentType="audio/mpeg"
        )
    
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.r2_bucket, "Key": object_key},
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
