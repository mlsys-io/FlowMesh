"""Stack env schema."""

from flowmesh_stack.env_schema import (
    EnvSchema,
    EnvSection,
    EnvVar,
    EnvVarType,
    require_all_or_none,
    require_if_true,
)

STACK_ENV_SCHEMA = EnvSchema(
    name="stack",
    header=[
        "# FlowMesh Stack Configuration",
        "# Copy to .env and adjust as needed",
    ],
    sections=[
        EnvSection(
            title="Image Source",
            vars=[
                EnvVar("FLOWMESH_REGISTRY", "ghcr.io/mlsys-io", required=True),
                EnvVar("FLOWMESH_VERSION", "dev", required=True),
                EnvVar("FLOWMESH_CACHE_VERSION"),
                EnvVar("FLOWMESH_BUILD_REF", "local"),
            ],
        ),
        EnvSection(
            title="Node Identity",
            vars=[
                EnvVar(
                    "NODE_ROLE",
                    "root",
                    var_type=EnvVarType.ENUM,
                    choices={"root", "worker"},
                ),
                EnvVar("NODE_NAMESPACE", "flowmesh"),
                EnvVar("NODE_CLUSTER", "dev"),
                EnvVar("NODE_ALIAS", "node"),
                EnvVar("NODE_TAGS", var_type=EnvVarType.CSV),
                EnvVar("ENABLE_SUPERVISOR", "true", var_type=EnvVarType.BOOL),
                EnvVar("SERVER_TOKEN", warn_if_empty=True),
                EnvVar("SERVER_HOST", "localhost", required=True),
                EnvVar(
                    "SERVER_HTTP_PORT",
                    "8000",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "SERVER_GRPC_PORT",
                    "50051",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "SERVER_LOG_LEVEL",
                    "INFO",
                    var_type=EnvVarType.LOG_LEVEL,
                ),
            ],
        ),
        EnvSection(
            title="Server gRPC TLS",
            description=["Leave empty to disable"],
            vars=[
                EnvVar(
                    "SERVER_TLS_DIR",
                    "./secrets/tls/server",
                    var_type=EnvVarType.DIR_PATH,
                    use_default=True,
                    ensure_path="create",
                ),
                EnvVar(
                    "SERVER_GRPC_TLS_CA_FILE",
                    "/etc/ssl/server/server-ca.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "SERVER_GRPC_TLS_CERT_FILE",
                    "/etc/ssl/server/server.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "SERVER_GRPC_TLS_KEY_FILE",
                    "/etc/ssl/server/server.key",
                    var_type=EnvVarType.FILE_PATH,
                ),
            ],
        ),
        EnvSection(
            title="Supervisor gRPC",
            description=[
                "Tuning for the supervisor's gRPC server and worker connections.",
                "Leave SUPERVISOR_GRPC_EXTERNAL_PORT empty unless workers connect",
                "through a port-forwarded / proxied address.",
            ],
            vars=[
                EnvVar(
                    "SUPERVISOR_GRPC_DISABLE_SERVER_TLS",
                    "false",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_EXTERNAL_PORT",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS",
                    "true",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_MIN_RECV_PING_INTERVAL_MS",
                    "60000",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_KEEPALIVE_TIME_MS",
                    "300000",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "SUPERVISOR_GRPC_KEEPALIVE_TIMEOUT_MS",
                    "10000",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
            ],
        ),
        EnvSection(
            title="Redis Connectivity",
            vars=[
                EnvVar(
                    "REDIS_CONTROL_URL",
                    "redis://localhost:6379/0",
                    var_type=EnvVarType.URL,
                    required=True,
                    url_schemes={"redis", "rediss"},
                ),
                EnvVar(
                    "REDIS_TELEMETRY_URL",
                    "redis://localhost:6380/0",
                    var_type=EnvVarType.URL,
                    required=True,
                    url_schemes={"redis", "rediss"},
                ),
            ],
        ),
        EnvSection(
            title="Core Ports",
            vars=[
                EnvVar(
                    "REDIS_CONTROL_PORT",
                    "6379",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "REDIS_TELEMETRY_PORT",
                    "6380",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="Log Streams (Redis)",
            description=["Caps Redis Streams for per-task and per-workflow logs."],
            vars=[
                EnvVar(
                    "LOG_STREAM_MAXLEN_TASK",
                    "50000",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "LOG_STREAM_MAXLEN_WORKFLOW",
                    "200000",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "LOG_STREAM_TTL_SEC",
                    "3600",
                    description="Expire log stream keys after close (0 disables).",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "TASK_LOG_ARCHIVE_FLUSH_INTERVAL_SEC",
                    "5",
                    description="Flush archived task logs at most every N seconds.",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.1,
                ),
                EnvVar(
                    "TASK_LOG_ARCHIVE_FLUSH_MAX_ENTRIES",
                    "100",
                    description="Flush archived task logs after buffering N entries.",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="Redis Access",
            vars=[
                EnvVar("REDIS_ACL_ENABLED", "1", var_type=EnvVarType.BOOL),
                EnvVar("REDIS_USERNAME", "admin"),
                EnvVar("REDIS_PASSWORD", "very-strong-password"),
            ],
        ),
        EnvSection(
            title="Redis TLS",
            description=["Leave empty to disable"],
            vars=[
                EnvVar(
                    "REDIS_TLS_DIR",
                    "./secrets/tls/redis",
                    var_type=EnvVarType.DIR_PATH,
                    use_default=True,
                    ensure_path="create",
                ),
                EnvVar(
                    "REDIS_TLS_CA_FILE",
                    "/etc/ssl/redis/redis-ca.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "REDIS_TLS_CERT_FILE",
                    "/etc/ssl/redis/redis-server.pem",
                    var_type=EnvVarType.FILE_PATH,
                ),
                EnvVar(
                    "REDIS_TLS_KEY_FILE",
                    "/etc/ssl/redis/redis-server.key",
                    var_type=EnvVarType.FILE_PATH,
                ),
            ],
        ),
        EnvSection(
            title="SSH Task Support",
            vars=[
                EnvVar("ENABLE_SERVER_SSH_PROXY", "true", var_type=EnvVarType.BOOL),
                EnvVar("ENABLE_SERVER_SSH_FORWARD", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "ENABLE_SERVER_SSH_CONNECTION_AUDIT",
                    "true",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar("SERVER_SSH_FORWARD_BIND_HOST", "0.0.0.0", required=True),
                EnvVar("SERVER_SSH_FORWARD_PUBLIC_HOST", "localhost", required=True),
                EnvVar(
                    "SERVER_SSH_FORWARD_PORT_START",
                    "32000",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
                EnvVar(
                    "SERVER_SSH_FORWARD_PORT_END",
                    "32100",
                    var_type=EnvVarType.INT,
                    required=True,
                    min_value=1,
                ),
            ],
        ),
        EnvSection(
            title="SSH Worker Defaults",
            vars=[
                EnvVar("ENABLE_SSH_BY_DEFAULT", "true", var_type=EnvVarType.BOOL),
                EnvVar("SSH_DEFAULT_IMAGE"),
                EnvVar("SSH_DEFAULT_USER"),
                EnvVar("SSH_DEFAULT_TTL_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_DEFAULT_IDLE_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_MAX_TTL_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_POLL_INTERVAL_SEC", var_type=EnvVarType.FLOAT, min_value=0),
                EnvVar("SSH_STOP_TIMEOUT_SEC", var_type=EnvVarType.FLOAT, min_value=0),
            ],
        ),
        EnvSection(
            title="General Settings",
            vars=[
                EnvVar("TZ", "Asia/Singapore", required=True),
                EnvVar(
                    "LOG_LEVEL", "INFO", var_type=EnvVarType.LOG_LEVEL, required=True
                ),
            ],
        ),
        EnvSection(
            title="Orchestrator Settings",
            vars=[
                EnvVar(
                    "ORCHESTRATOR_DISPATCH_MODE",
                    "adaptive",
                    var_type=EnvVarType.ENUM,
                    choices={"adaptive"},
                ),
                EnvVar(
                    "ORCHESTRATOR_WORKER_SELECTION",
                    "best_fit",
                    var_type=EnvVarType.ENUM,
                    choices={"best_fit", "first_fit", "min_satisfying"},
                ),
                EnvVar(
                    "SCHEDULER_SELECTION_JITTER",
                    "0.001",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SCHEDULER_LAMBDA_INFERENCE",
                    "0.4",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SCHEDULER_LAMBDA_TRAINING",
                    "0.8",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SCHEDULER_LAMBDA_OTHER",
                    "0.5",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar("ENABLE_TASK_MERGE", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "TASK_MERGE_MAX_BATCH_SIZE",
                    "4",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar("ENABLE_CONTEXT_REUSE", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "WORKER_CACHE_TTL_SEC",
                    "3600",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar(
                    "ENABLE_STAGE_WEIGHT_STICKINESS",
                    "false",
                    var_type=EnvVarType.BOOL,
                ),
                EnvVar("ENABLE_WORKER_WATCHDOG", "true", var_type=EnvVarType.BOOL),
                EnvVar(
                    "WORKER_DEATH_CHECK_INTERVAL",
                    "30",
                    var_type=EnvVarType.INT,
                    min_value=5,
                ),
                EnvVar(
                    "WORKER_DEATH_GRACE_SEC",
                    "60",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
            ],
        ),
        EnvSection(
            title="Server Heartbeat",
            vars=[
                EnvVar(
                    "SERVER_HEARTBEAT_INTERVAL",
                    "30",
                    var_type=EnvVarType.INT,
                    min_value=1,
                )
            ],
        ),
        EnvSection(
            title="Vast.ai Configuration",
            vars=[
                EnvVar("VAST_SEARCH_LIMIT", var_type=EnvVarType.INT, min_value=0),
                EnvVar("VAST_MAX_RETRIES", var_type=EnvVarType.INT, min_value=0),
            ],
        ),
        EnvSection(
            title="Worker Parameters",
            vars=[
                EnvVar("WORKER_LOG_LEVEL", "INFO", var_type=EnvVarType.LOG_LEVEL),
                EnvVar(
                    "HEARTBEAT_INTERVAL_SEC", "30", var_type=EnvVarType.INT, min_value=1
                ),
                EnvVar(
                    "WORKER_COST_PER_HOUR",
                    "1.0",
                    var_type=EnvVarType.FLOAT,
                    min_value=0.0,
                ),
                EnvVar(
                    "SERVER_RESULTS_DIR",
                    "flowmesh_results",
                    var_type=EnvVarType.DIR_PATH,
                    description="Set to the same value as WORKER_RESULTS_DIR to enable "
                    "server-side access to worker results.",
                ),
                EnvVar(
                    "WORKER_RESULTS_DIR",
                    "flowmesh_results",
                    var_type=EnvVarType.DIR_PATH,
                ),
                EnvVar("HF_CACHE_DIR", var_type=EnvVarType.DIR_PATH),
                EnvVar(
                    "WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar("WORKER_TAGS", var_type=EnvVarType.CSV),
                EnvVar("WORKER_HB_DIR", var_type=EnvVarType.DIR_PATH),
                EnvVar(
                    "FLOWMESH_BASE_URL",
                    "http://localhost:8000",
                    var_type=EnvVarType.URL,
                    required=True,
                    url_schemes={"http", "https"},
                ),
                EnvVar(
                    "NEBULA_API_BASE_URL",
                    var_type=EnvVarType.URL,
                    url_schemes={"http", "https"},
                ),
                EnvVar(
                    "CUDA_VISIBLE_DEVICES", "all", var_type=EnvVarType.CSV_INTS_OR_ALL
                ),
                EnvVar("WORKER_UPLOAD_RESULTS", "false", var_type=EnvVarType.BOOL),
                EnvVar(
                    "MODEL_CLEANUP_AFTER_UPLOAD",
                    "0",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
            ],
        ),
        EnvSection(
            title="Model Pre-downloading",
            description=[
                "Comma-separated list of models to pre-download during worker startup",
                "Leave empty to disable model pre-downloading",
                "Example: meta-llama/Llama-3.2-1B-Instruct,"
                "meta-llama/Llama-3.2-3B-Instruct",
            ],
            vars=[EnvVar("PREDOWNLOAD_MODEL_LIST", var_type=EnvVarType.CSV)],
        ),
        EnvSection(
            title="API Keys injected into workers (optional)",
            vars=[
                EnvVar("OPENAI_API_KEY"),
                EnvVar("GOOGLE_API_KEY"),
                EnvVar("VAST_API_KEY"),
                EnvVar("HF_TOKEN"),
                EnvVar("NEBULA_API_TOKEN"),
            ],
        ),
        EnvSection(
            title="External Plugins",
            description=[
                "# Comma-separated module names imported at server startup. Each ",
                "# named module must expose `install()`, which registers entries ",
                "# into server.hooks.IDENTITY_PROVIDERS / SUBMISSION_GUARDS / ",
                "# USAGE_SINKS. Leave empty in OSS unless you ship a plugin.",
            ],
            vars=[EnvVar("FLOWMESH_PLUGINS", "")],
        ),
        EnvSection(
            title="Agent Executor (youtu-agent / utu)",
            description=[
                "All four UTU_LLM_* are required for the agent executor to run."
            ],
            vars=[
                EnvVar(
                    "UTU_LLM_TYPE",
                    description='utu LLM provider kind, e.g. "chat.completions"',
                ),
                EnvVar(
                    "UTU_LLM_MODEL",
                    description="utu model identifier, e.g. gpt-4o-mini",
                ),
                EnvVar(
                    "UTU_LLM_BASE_URL",
                    description="utu LLM base URL",
                    var_type=EnvVarType.URL,
                    url_schemes={"http", "https"},
                ),
                EnvVar(
                    "UTU_LLM_API_KEY",
                    description="utu LLM API key",
                ),
                EnvVar(
                    "SERPER_API_KEY",
                    description="Serper API key (optional, for agent search tools)",
                ),
                EnvVar(
                    "JINA_API_KEY",
                    description="Jina API key (optional, for agent search tools)",
                ),
                EnvVar(
                    "DB_URL",
                    description="Database URL for agent tracing (optional)",
                ),
            ],
        ),
        EnvSection(
            title="n8n Integration",
            vars=[
                EnvVar(
                    "N8N_CREDENTIAL_AES_PASSWORD",
                    description="AES-GCM key to decrypt encrypted n8n credentials.",
                    warn_if_empty=True,
                )
            ],
        ),
        EnvSection(
            title="Worker launch config (optional)",
            vars=[
                EnvVar(
                    "SERVER_WORKER_CONFIG",
                    "./configs/worker_config.yaml",
                    var_type=EnvVarType.FILE_PATH,
                    use_default=True,
                    ensure_path="create",
                )
            ],
        ),
        EnvSection(
            title="Logging",
            vars=[
                EnvVar(
                    "LOG_MAX_BYTES",
                    "5242880",
                    var_type=EnvVarType.INT,
                    min_value=1,
                ),
                EnvVar(
                    "LOG_BACKUP_COUNT",
                    "5",
                    var_type=EnvVarType.INT,
                    min_value=0,
                ),
                EnvVar("SERVER_APP_RELOAD", "0", var_type=EnvVarType.BOOL),
                EnvVar("SERVER_APP_LOG_LEVEL", "info", var_type=EnvVarType.LOG_LEVEL),
            ],
        ),
    ],
    validators=[
        lambda env, errors, warnings: require_if_true(
            env, "REDIS_ACL_ENABLED", ["REDIS_USERNAME", "REDIS_PASSWORD"], errors
        ),
        lambda env, errors, warnings: require_all_or_none(
            env,
            [
                "SERVER_GRPC_TLS_CA_FILE",
                "SERVER_GRPC_TLS_CERT_FILE",
                "SERVER_GRPC_TLS_KEY_FILE",
            ],
            errors,
        ),
    ],
)
