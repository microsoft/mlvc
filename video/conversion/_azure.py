# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.ml import MLClient
from ._env import get_required_env


def download_blob(
    account_name: str,
    container_name: str,
    blob_name: Path | str,
    destination_dir: Path | str,
    overwrite: bool = True,
) -> Path:
    destination_file_path = Path(destination_dir) / Path(blob_name)
    if not overwrite and destination_file_path.exists():
        # print(f"File {destination_file_path} already exists, skipping download.")
        return destination_file_path

    print(f"Downloading {blob_name} from {account_name}/{container_name} to {destination_file_path}...")

    # Connect
    download_start = time.time()
    credential = DefaultAzureCredential()
    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential,
    )

    # Download
    destination_file_path.parent.mkdir(parents=True, exist_ok=True)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=str(blob_name))
    download_stream = blob_client.download_blob(max_concurrency=4)
    with open(destination_file_path, "wb") as download_file:
        for chunk in download_stream.chunks():
            download_file.write(chunk)

    # Print stats
    download_duration = time.time() - download_start
    download_speed = download_stream.size / download_duration / 1024**2
    print(
        f"Downloaded {blob_name} ({download_stream.size / 1024**2:.3f} MB) "
        f"to {destination_file_path} in {download_duration:.3f} seconds ({download_speed:.3f} MB/s)"
    )
    return destination_file_path


def get_azureml_job(job_name: str):
    print(f"Getting Azure ML job {job_name}...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential,
        subscription_id=get_required_env("AZURE_ML_SUBSCRIPTION_ID"),
        resource_group_name=get_required_env("AZURE_ML_RESOURCE_GROUP"),
        workspace_name=get_required_env("AZURE_ML_WORKSPACE"),
    )
    job = ml_client.jobs.get(job_name)
    print(f"Got Azure ML job {job.display_name}.")
    return job


if __name__ == "__main__":
    pass
