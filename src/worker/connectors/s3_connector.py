"""S3/MinIO connector using boto3."""

import io
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import pandas as pd
from PIL import Image

try:
    import boto3
    from botocore.client import BaseClient, Config
    from botocore.exceptions import ClientError

    _HAS_BOTO3 = True
except ImportError:
    if TYPE_CHECKING:
        import boto3
        from botocore.client import BaseClient, Config
        from botocore.exceptions import ClientError

    _HAS_BOTO3 = False

from .base_connector import BaseConnector, ConnectorError, ConnectorResult

logger = logging.getLogger(__name__)


class S3Connector(BaseConnector):
    """S3/MinIO connector using boto3.

    Configuration:
        - connection_string: S3 connection string (required)
          Format: "s3://access_key:secret_key@endpoint:port/bucket_name/prefix"
          Example: "s3://minioadmin:minioadmin@localhost:9000/mybucket/data"
        - cert_data: PEM-encoded CA certificate data as string (optional, for
                     self-signed certs). When provided, HTTPS is used and the
                     certificate is used for verification.

    Example config:
        {
            "connection_string": "s3://access:secret@minio.example.com:9000/data/prefix",
            "cert_data": "-----BEGIN CERTIFICATE-----\\n...\\n-----END CERTIFICATE-----"
        }
    """

    name = "s3"

    def __init__(
        self,
        connection_string: str,
        cert_data: str | None = None,
    ):
        if not _HAS_BOTO3:
            raise ConnectorError(
                "boto3 is required for S3Connector. Install with: pip install boto3"
            )

        self._connection_string = connection_string
        self._cert_data = cert_data
        self._s3_client: BaseClient | None = None
        self._bucket_name: str | None = None
        self._key_prefix: str | None = None
        self._endpoint_url: str | None = None
        self._access_key: str | None = None
        self._secret_key: str | None = None
        self._cert_file_path: str | None = None

        self._parse_connection_string()

    def _parse_connection_string(self) -> None:
        """Parse S3 connection string into components."""
        # Format: s3://access_key:secret_key@endpoint:port/bucket_name/prefix
        try:
            parsed = urlparse(self._connection_string)

            if parsed.scheme != "s3":
                raise ValueError(f"Invalid scheme: {parsed.scheme}. Expected s3")

            # Extract credentials
            if parsed.username and parsed.password:
                self._access_key = parsed.username
                self._secret_key = parsed.password
            else:
                raise ValueError(
                    "Access key and secret key are required in connection string"
                )

            # Extract endpoint
            if parsed.hostname:
                protocol = "https" if self._cert_data else "http"
                port_str = f":{parsed.port}" if parsed.port else ""
                self._endpoint_url = f"{protocol}://{parsed.hostname}{port_str}"
            else:
                raise ValueError("Endpoint hostname is required in connection string")

            # Extract bucket name from path
            if parsed.path and parsed.path.strip("/"):
                path_parts = parsed.path.strip("/").split("/")
                self._bucket_name = path_parts[0]
                prefix = "/".join(path_parts[1:]).strip("/")
                self._key_prefix = prefix
            else:
                raise ValueError("Bucket name is required in connection string path")

            logger.info(
                "Parsed S3 connection string: endpoint=%s, bucket=%s",
                self._endpoint_url,
                self._bucket_name,
            )

        except Exception as e:
            raise ConnectorError(f"Failed to parse S3 connection string: {e}") from e

    def _write_cert_to_temp_file(self) -> str | None:
        """Write certificate data to a temporary file if provided."""
        if not self._cert_data:
            return None

        import tempfile

        try:
            # Create a named temporary file that persists
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pem", delete=False, prefix="s3_ca_"
            ) as cert_file:
                cert_file.write(self._cert_data)
                cert_path = cert_file.name

            logger.info("Wrote CA certificate to temporary file: %s", cert_path)
            return cert_path

        except Exception as e:
            logger.warning("Failed to write certificate to temp file: %s", e)
            return None

    def _apply_prefix(self, key: str) -> str:
        normalized_key = key.lstrip("/")
        if not self._key_prefix:
            return normalized_key
        return f"{self._key_prefix}/{normalized_key}"

    def connect(self) -> None:
        """Establish connection to S3/MinIO."""
        if self._s3_client is not None:
            return  # Already connected

        try:
            if not self._endpoint_url or not self._access_key or not self._secret_key:
                raise ConnectorError("S3 connection is not fully configured")

            client_kwargs: dict[str, Any] = {
                "endpoint_url": self._endpoint_url,
                "aws_access_key_id": self._access_key,
                "aws_secret_access_key": self._secret_key,
                "config": Config(signature_version="s3v4"),
            }

            # If certificate data is provided, write it to a temp file
            if self._cert_data:
                self._cert_file_path = self._write_cert_to_temp_file()
                if self._cert_file_path:
                    client_kwargs["verify"] = self._cert_file_path

            # Create boto3 client with custom configuration
            assert boto3 is not None
            self._s3_client = boto3.client("s3", **client_kwargs)

            # Test connection by listing objects (limited to 1)
            self._s3_client.list_objects_v2(  # type: ignore
                Bucket=self._bucket_name, MaxKeys=1
            )

            logger.info(
                "Connected to S3: endpoint=%s, bucket=%s",
                self._endpoint_url,
                self._bucket_name,
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise ConnectorError(
                f"Failed to connect to S3 (Error: {error_code}): {e}"
            ) from e
        except Exception as e:
            raise ConnectorError(f"Failed to connect to S3: {e}") from e

    def disconnect(self) -> None:
        """Close S3 connection and cleanup temporary files."""
        if self._s3_client is not None:
            try:
                # boto3 clients don't need explicit closing
                self._s3_client = None
                logger.info("Disconnected from S3")
            except Exception as e:
                logger.warning("Error during S3 disconnect: %s", e)

        # Clean up temporary certificate file
        if self._cert_file_path:
            try:
                import os

                os.unlink(self._cert_file_path)
                logger.info(
                    "Removed temporary certificate file: %s", self._cert_file_path
                )
                self._cert_file_path = None
            except Exception as e:
                logger.warning("Failed to remove temporary certificate file: %s", e)

    def execute(
        self,
        query: str | list[str],
        *args: Any,
        encoding: str = "utf-8",
        as_dataframe: bool = False,
        **_,
    ) -> ConnectorResult:
        """Retrieve files from S3/MinIO.

        Args:
            query: Single file key (str) or list of file keys (list[str]) to retrieve
                   File keys are relative to the bucket root or the configured
                   prefix in the connection string.
                   Example: "data/file1.csv" or ["data/file1.csv", "data/file2.json"]
            **kwargs: Additional options:
                - encoding: Text encoding for file contents (default: 'utf-8')
                - as_dataframe: If True and file is CSV, return as DataFrame
                                (default: False)

        Returns:
            Dict with:
                - success: bool
                - data: pandas Series with file keys as index and contents as values
                       For single file, Series with one element
                       For multiple files, Series with multiple elements
                - error: Error message if failed
                - metadata: File information (size, content_type, etc.)
        """
        if self._s3_client is None:
            self.connect()

        start_time = time.time()
        result: ConnectorResult = {
            "success": False,
            "data": None,
            "error": None,
            "metadata": {},
        }

        try:
            # Handle both single file and list of files
            if isinstance(query, str):
                file_keys = [query]
            elif isinstance(query, list):
                file_keys = query
            else:
                raise ValueError(f"query must be str or list[str], got {type(query)}")

            if not file_keys:
                raise ValueError("At least one file key must be provided")

            # Retrieve all files
            file_contents: dict[str, pd.DataFrame | Image.Image | str | None] = {}
            file_metadata = {}

            for file_key in file_keys:
                try:
                    full_key = self._apply_prefix(file_key)
                    # Get object from S3
                    response = self._s3_client.get_object(  # type: ignore
                        Bucket=self._bucket_name, Key=full_key
                    )

                    # Read content
                    content_bytes: bytes = response["Body"].read()
                    content_type: str = response.get("ContentType") or ""

                    # Store metadata
                    file_metadata[file_key] = {
                        "size": response["ContentLength"],
                        "content_type": content_type,
                        "last_modified": response["LastModified"].isoformat(),
                        "etag": response["ETag"],
                    }

                    # Decode content
                    content: pd.DataFrame | Image.Image | str
                    if as_dataframe and file_key.lower().endswith(".csv"):
                        # Parse as CSV DataFrame
                        content = pd.read_csv(io.BytesIO(content_bytes))
                    elif content_type.startswith("image/"):
                        content = Image.open(io.BytesIO(content_bytes)).convert("RGB")
                    else:
                        content = content_bytes.decode(encoding)

                    file_contents[file_key] = content

                    logger.debug(
                        "Retrieved file: %s (size: %d bytes)",
                        full_key,
                        response["ContentLength"],
                    )

                except ClientError as e:
                    error_code = e.response.get("Error", {}).get("Code", "Unknown")
                    if error_code == "NoSuchKey":
                        logger.warning("File not found: %s", file_key)
                        file_contents[file_key] = None
                        file_metadata[file_key] = {"error": "File not found"}
                    else:
                        raise

            # Convert to pandas Series
            data_series = pd.Series(file_contents)

            execution_time = time.time() - start_time

            result.update(
                {
                    "success": True,
                    "data": data_series,
                    "metadata": {
                        "execution_time": execution_time,
                        "file_count": len(file_keys),
                        "files": file_metadata,
                    },
                }
            )

            logger.info(
                "Retrieved %d file(s) from S3 in %.3fs", len(file_keys), execution_time
            )

        except Exception as e:
            error_msg = f"Failed to retrieve files from S3: {e}"
            result["error"] = error_msg
            logger.error(error_msg, exc_info=True)

        return result

    def get_schema(self, table_name: str | None = None) -> dict[str, Any]:
        """List files/prefixes in the S3 bucket.

        Args:
            table_name: Optional prefix to filter files (like a "folder" path)
                       This is applied after any prefix from the connection string.
                       If None, lists all objects in the bucket/prefix

        Returns:
            Dict with:
                - bucket: Bucket name
                - prefix: The prefix used for listing
                - objects: List of object keys
                - prefixes: List of common prefixes (folders)
        """
        if self._s3_client is None:
            self.connect()
            assert self._s3_client is not None

        try:
            base_prefix = self._key_prefix or ""
            if table_name:
                table_prefix = table_name.lstrip("/")
                prefix = (
                    f"{base_prefix}/{table_prefix}" if base_prefix else table_prefix
                )
            else:
                prefix = base_prefix

            # List objects with optional prefix
            paginator = self._s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self._bucket_name, Prefix=prefix, Delimiter="/"
            )

            objects = []
            prefixes = []

            for page in pages:
                # Files
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects.append(
                            {
                                "key": obj["Key"],
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"].isoformat(),
                                "etag": obj["ETag"],
                            }
                        )

                # Folders/prefixes
                if "CommonPrefixes" in page:
                    for prefix_obj in page["CommonPrefixes"]:
                        prefixes.append(prefix_obj["Prefix"])

            return {
                "bucket": self._bucket_name,
                "prefix": prefix,
                "objects": objects,
                "prefixes": prefixes,
                "object_count": len(objects),
                "prefix_count": len(prefixes),
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            raise ConnectorError(
                f"Failed to list S3 objects ({error_code}): {e}"
            ) from e
        except Exception as e:
            raise ConnectorError(f"Failed to list S3 objects: {e}") from e
