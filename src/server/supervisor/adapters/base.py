import os
from abc import ABC, abstractmethod
from typing import NewType, Self

from pydantic import BaseModel, ConfigDict, SecretStr

from ... import env
from ..schemas import WorkerInfo, WorkerStatus
from .utils import env_to_secret_str, to_env_str


class WorkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    supervisor_grpc_target: str = f"{env.SERVER_LOCAL_HOST}:{env.SERVER_GRPC_PORT}"
    """Supervisor gRPC target"""
    results_dir: str = env.WORKER_RESULTS_DIR
    """Root directory for task outputs"""
    hb_interval: int = env.SERVER_HEARTBEAT_INTERVAL
    """Interval between heartbeats in seconds"""
    worker_alias: str | None = None
    """Optional worker alias"""
    tags: str = env.WORKER_TAGS
    """Comma-separated tags used by the scheduler"""
    hb_file: str | None = None
    """Path to the worker heartbeat file."""
    log_level: str = env.LOG_LEVEL
    """Logging level for the worker"""
    flowmesh_url: str = env.FLOWMESH_BASE_URL
    """FlowMesh HTTP base URL required to build artifact download links"""
    worker_cost_per_hour: float | None = None
    """Hourly cost in USD"""
    model_archive_use_pigz: bool | None = None
    """Whether to use pigz for model archive compression"""
    model_archive_compression_level: int | None = None
    """Gzip compression level (0-9)"""
    model_archive_pigz_threads: int | None = None
    """Number of threads for pigz compression"""
    model_archive_pigz_bin: str | None = None
    """Path to pigz binary"""
    model_archive_tar_bin: str | None = None
    """Path to tar binary"""
    network_bandwidth: float | None = None
    """Bandwidth in bytes per second to throttle HTTP uploads"""
    openai_api_key: SecretStr | None = env_to_secret_str("OPENAI_API_KEY")
    """OpenAI API key"""
    google_api_key: SecretStr | None = env_to_secret_str("GOOGLE_API_KEY")
    """Google API key"""
    hf_token: SecretStr | None = env_to_secret_str("HF_TOKEN")
    """Hugging Face API token"""
    hf_cache_dir: str | None = env.HF_CACHE_DIR
    """Hugging Face cache directory"""
    predownload_model_list: str = env.PREDOWNLOAD_MODEL_LIST
    """Comma-separated list of models to pre-download during worker startup"""
    nebula_api_token: SecretStr | None = env_to_secret_str("NEBULA_API_TOKEN")
    """Nebula API token"""
    utu_llm_type: str | None = os.getenv("UTU_LLM_TYPE")
    """Agent executor (utu) LLM provider kind, e.g. "chat.completions" """
    utu_llm_model: str | None = os.getenv("UTU_LLM_MODEL")
    """Agent executor (utu) model identifier"""
    utu_llm_base_url: str | None = os.getenv("UTU_LLM_BASE_URL")
    """Agent executor (utu) LLM base URL"""
    utu_llm_api_key: SecretStr | None = env_to_secret_str("UTU_LLM_API_KEY")
    """Agent executor (utu) LLM API key"""
    serper_api_key: SecretStr | None = env_to_secret_str("SERPER_API_KEY")
    """Serper API key for agent search tools"""
    jina_api_key: SecretStr | None = env_to_secret_str("JINA_API_KEY")
    """Jina API key for agent search tools"""
    db_url: SecretStr | None = env_to_secret_str("DB_URL")
    """Database URL for agent tracing (may contain credentials)"""
    upload_results: bool = env.WORKER_UPLOAD_RESULTS
    """Whether to always upload results to the server if spec.output.destination
    is unspecified."""


WorkerTokenType = NewType("WorkerTokenType", str)


class WorkerAdapter(ABC):
    def __init__(self, token: WorkerTokenType, name: str, config: WorkerConfig) -> None:
        self._worker_id: str | None = None
        self.token = token
        self.name = name
        self.config = config

    @property
    @abstractmethod
    def status(self) -> WorkerStatus:
        pass

    @property
    def worker_id(self) -> str | None:
        return self._worker_id

    @abstractmethod
    def set_status(self, status: WorkerStatus) -> None:
        pass

    def set_worker_id(self, worker_id: str) -> None:
        if self._worker_id is not None:
            raise RuntimeError(f"Worker ID is already set to {self._worker_id}")
        self._worker_id = worker_id

    def clear_worker_id(self) -> None:
        if self._worker_id is None:
            raise RuntimeError("Worker ID is not set")
        self._worker_id = None

    @abstractmethod
    def get_info(self) -> WorkerInfo:
        pass

    @abstractmethod
    async def start(self) -> bool:
        """Start worker. Returns whether the worker was successfully started."""
        pass

    async def prepare(self) -> None:
        """Prepare worker (e.g., collecting hardware information) without starting
        it."""
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """Stop worker. Returns whether the worker was successfully stopped."""
        pass

    def _base_environment(self) -> dict[str, str]:
        config = self.config
        hb_file = config.hb_file or os.path.join(env.WORKER_HB_DIR, f"{self.token}.hb")
        return {
            "WORKER_TOKEN": self.token,  # type: ignore
            "SUPERVISOR_GRPC_TARGET": config.supervisor_grpc_target,
            "SUPERVISOR_GRPC_TLS_CA_B64": env.SERVER_GRPC_TLS_CA_B64,
            "RESULTS_DIR": config.results_dir,
            "HEARTBEAT_INTERVAL_SEC": to_env_str(config.hb_interval),
            "WORKER_HB_FILE": hb_file,
            "WORKER_NAMESPACE": env.NODE_NAMESPACE,
            "WORKER_CLUSTER": env.NODE_CLUSTER,
            "WORKER_ALIAS": config.worker_alias or "",
            "WORKER_TAGS": config.tags,
            "LOG_LEVEL": config.log_level,
            "WORKER_COST_PER_HOUR": to_env_str(config.worker_cost_per_hour),
            "FLOWMESH_BASE_URL": config.flowmesh_url,
            "MODEL_ARCHIVE_USE_PIGZ": to_env_str(config.model_archive_use_pigz),
            "MODEL_ARCHIVE_COMPRESSION_LEVEL": to_env_str(
                config.model_archive_compression_level
            ),
            "MODEL_ARCHIVE_PIGZ_THREADS": to_env_str(config.model_archive_pigz_threads),
            "MODEL_ARCHIVE_PIGZ_BIN": to_env_str(config.model_archive_pigz_bin),
            "MODEL_ARCHIVE_TAR_BIN": to_env_str(config.model_archive_tar_bin),
            "WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC": to_env_str(
                config.network_bandwidth
            ),
            "WORKER_UPLOAD_RESULTS": to_env_str(config.upload_results),
            "DOCKER_GPU_RUNTIME": to_env_str(env.DOCKER_GPU_RUNTIME),
            "FLOWMESH_API_KEY": to_env_str(env.FLOWMESH_API_KEY),
            "OPENAI_API_KEY": to_env_str(config.openai_api_key),
            "GOOGLE_API_KEY": to_env_str(config.google_api_key),
            "HF_TOKEN": to_env_str(config.hf_token),
            "PREDOWNLOAD_MODEL_LIST": config.predownload_model_list,
            "NEBULA_API_TOKEN": to_env_str(config.nebula_api_token),
            "NEBULA_API_BASE_URL": env.NEBULA_API_BASE_URL,
            "UTU_LLM_TYPE": to_env_str(config.utu_llm_type),
            "UTU_LLM_MODEL": to_env_str(config.utu_llm_model),
            "UTU_LLM_BASE_URL": to_env_str(config.utu_llm_base_url),
            "UTU_LLM_API_KEY": to_env_str(config.utu_llm_api_key),
            "SERPER_API_KEY": to_env_str(config.serper_api_key),
            "JINA_API_KEY": to_env_str(config.jina_api_key),
            "DB_URL": to_env_str(config.db_url),
        }


class WorkerFactory(ABC):
    _instance: Self | None = None

    @classmethod
    def get_instance(cls) -> Self:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @abstractmethod
    def create_worker(self, token: WorkerTokenType, *args, **kwargs) -> WorkerAdapter:
        pass

    @abstractmethod
    def destroy_worker(self, worker: WorkerAdapter) -> None:
        pass

    def cleanup(self) -> None:
        pass
