from pathlib import PurePosixPath

from pydantic import BaseModel, Field


class ResultDeliveryRequest(BaseModel):
    all_artifacts: bool = False
    artifact_fields: list[str] = Field(default_factory=list)


class ArtifactInput(BaseModel):
    task_id: str
    path: str
    source: str
    generation: str | None = None


class DeliveredFile(BaseModel):
    size: int
    sha256: str


class ResultDeliveryReceipt(BaseModel):
    task_id: str
    generation: str
    all_artifacts: bool = False
    artifact_paths: list[str] = Field(default_factory=list)
    directories: list[str] = Field(default_factory=list)
    files: dict[str, DeliveredFile] = Field(default_factory=dict)
    symlinks: dict[str, str] = Field(default_factory=dict)

    def has_artifact(self, selection: str) -> bool:
        name = PurePosixPath("artifacts", selection).as_posix()
        return name in self.directories or name in self.files or name in self.symlinks
