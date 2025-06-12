import argparse
import json
from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
from prefect_gcp import GcsBucket, GcpCredentials
from prefect.blocks.core import Block


def create_aws_creds_block():
    aws_creds = AwsCredentials(
        aws_access_key_id="123abc",
        aws_secret_access_key="abc123"
    )
    aws_creds.save(name="my-aws-creds", overwrite=True)
    print("✅ AWS credentials block created: my-aws-creds")


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    s3_bucket = S3Bucket(
        bucket_name="my-first-bucket-abc",
        credentials=aws_creds
    )
    s3_bucket.save(name="s3-bucket-example", overwrite=True)
    print("✅ S3 bucket block created: s3-bucket-example")


def create_gcp_creds_block(service_account_file: str):
    with open(service_account_file, "r") as f:
        service_account_info = json.load(f)
    #print(type(service_account_info))
    #print(service_account_info)

    gcp_creds = GcpCredentials(service_account_info=service_account_info)
    gcp_creds.save(name="my-gcp-creds", overwrite=True)
    print("✅ GCP credentials block created: my-gcp-creds")


def create_gcs_bucket_block(bucket_name: str):
    gcp_creds = GcpCredentials.load("my-gcp-creds")
    if gcp_creds.service_account_info:
        # Unwrap the secret dictionary
        service_account_info = gcp_creds.service_account_info.get_secret_value()
        print("🔐 Loaded GCP service account info:")
        #print(json.dumps(service_account_info, indent=2))
    else:
        print("⚠️ No service_account_info found in this credentials block.")
    gcs_bucket = GcsBucket(
        bucket=bucket_name,
        gcp_credentials=gcp_creds
    )
    gcs_bucket.save(name="gcs-bucket-example", overwrite=True)
    print("✅ GCS bucket block created: gcs-bucket-example")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Prefect cloud storage blocks for AWS or GCP.")
    parser.add_argument("--cloud", choices=["aws", "gcp"], required=True, help="Cloud provider: aws or gcp")
    parser.add_argument("--gcp-key", type=str, help="Path to GCP service account JSON file")
    parser.add_argument("--gcp-bucket", type=str, help="GCS bucket name")

    args = parser.parse_args()

    if args.cloud == "aws":
        create_aws_creds_block()
        sleep(3)
        create_s3_bucket_block()

    elif args.cloud == "gcp":
        if not args.gcp_key or not args.gcp_bucket:
            parser.error("For GCP, both --gcp-key and --gcp-bucket must be provided.")
        create_gcp_creds_block(args.gcp_key)
        sleep(3)
        create_gcs_bucket_block(args.gcp_bucket)

# python 3.5/create_aws_or_gcp_block.py --cloud aws
# python 3.5/create_aws_or_gcp_block.py --cloud gcp --gcp-key "/home/habeeb/dprof-dezoomfinal-b4d188529d18.json" --gcp-bucket  prefect-zoomcamp25