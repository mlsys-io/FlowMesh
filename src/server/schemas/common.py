from pydantic import BaseModel, Field


class OkResponse(BaseModel):
    ok: bool = Field(description="Whether the request succeeded.")


class VersionResponse(BaseModel):
    version: str = Field(description="Server version.")


class PathResponse(OkResponse):
    path: str = Field(description="Filesystem path of the stored artifact.")
