import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aioboto3
from botocore.exceptions import ClientError

BUCKETS: tuple[str, ...] = (
    "originals",
    "protected",
    "datasets",
    "checkpoints",
    "evaluation",
)


class ObjectStorage:
    def __init__(self) -> None:
        self._session = aioboto3.Session()
        self._client_config: dict[str, Any] = {
            "service_name": "s3",
            "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
            "aws_access_key_id": os.getenv("S3_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("S3_SECRET_ACCESS_KEY"),
            "region_name": os.getenv("S3_REGION", "us-east-1"),
        }

    @asynccontextmanager
    async def client(self) -> AsyncIterator[Any]:
        async with self._session.client(**self._client_config) as s3:
            yield s3

    async def ensure_buckets(self) -> None:
        async with self.client() as s3:
            for bucket in BUCKETS:
                try:
                    await s3.head_bucket(Bucket=bucket)
                except ClientError as exc:
                    status = exc.response.get("ResponseMetadata", {}).get(
                        "HTTPStatusCode"
                    )
                    code = exc.response.get("Error", {}).get("Code")
                    if status != 404 and code not in {"404", "NoSuchBucket"}:
                        raise

                    region = self._client_config["region_name"]
                    request: dict[str, Any] = {"Bucket": bucket}
                    if region != "us-east-1":
                        request["CreateBucketConfiguration"] = {
                            "LocationConstraint": region
                        }
                    await s3.create_bucket(**request)

    async def upload_file(
        self,
        *,
        bucket: str,
        key: str,
        filename: str,
        content_type: str | None = None,
    ) -> None:
        extra_args: dict[str, str] = {}
        if content_type is not None:
            extra_args["ContentType"] = content_type

        async with self.client() as s3:
            await s3.upload_file(
                filename,
                bucket,
                key,
                ExtraArgs=extra_args or None,
            )

    async def download_file(
        self,
        *,
        bucket: str,
        key: str,
        filename: str,
    ) -> None:
        async with self.client() as s3:
            await s3.download_file(bucket, key, filename)

    async def put_object(
        self,
        *,
        bucket: str,
        key: str,
        body: bytes,
        content_type: str | None = None,
    ) -> None:
        request: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": body}
        if content_type is not None:
            request["ContentType"] = content_type

        async with self.client() as s3:
            await s3.put_object(**request)

    async def get_object(
        self,
        *,
        bucket: str,
        key: str,
    ) -> bytes:
        async with self.client() as s3:
            response = await s3.get_object(Bucket=bucket, Key=key)
            async with response["Body"] as stream:
                return await stream.read()


storage = ObjectStorage()
