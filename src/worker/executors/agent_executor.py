#!/usr/bin/env python3
"""
Agent Executor

Integrates youtu-agent framework into FlowMesh system.
Supports task-based agent execution with streaming output and progress tracking.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

from shared.schemas.artifact import ArtifactRef
from shared.schemas.result import BaseExecutorResult
from shared.tasks.specs import AgentSpecStrict

from .base_executor import ExecutionError, Executor, ExecutorTask
from .utils.checkpoints import maybe_upload_artifacts, write_executor_result
from .utils.graph_templates import build_prompts_from_graph_template

# Add agent directory to sys.path for utu imports
agent_dir = Path(__file__).parent / "agent"
sys.path.insert(0, str(agent_dir))

# Constants
DEFAULT_CONFIG_NAME = "default"
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_DATASET_COLUMN = "text"
DEFAULT_AGENT_TASK_TIMEOUT_SEC = 600


def _resolve_task_timeout(agent: dict[str, Any] | None) -> int:
    if not agent:
        return DEFAULT_AGENT_TASK_TIMEOUT_SEC
    raw = agent.get("timeout")
    if raw is None:
        return DEFAULT_AGENT_TASK_TIMEOUT_SEC
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise ExecutionError(
            f"spec.agent.timeout must be a positive integer (seconds), got {raw!r}"
        )
    return raw


logger = logging.getLogger("worker.agent")


class AgentResult(BaseExecutorResult):
    ok: bool = True
    model: str
    items: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    agent_output: ArtifactRef | None = None
    batch_summary_file: ArtifactRef | None = None


class AgentExecutor(Executor):
    """Agent executor using youtu-agent (utu) framework"""

    name = "agent"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._initialized = False
        self._tasks: list[str] = []

    def prepare(self) -> None:
        """Initialize utu (youtu-agent) dependencies"""
        if self._initialized:
            return

        try:
            # Load secrets from secrets.yaml if environment variables are not set
            self._setup_environment_from_secrets()

            # Import utu modules from the correct path
            from utu.agents import get_agent  # type: ignore # noqa: F401
            from utu.config import (  # type: ignore # noqa: F401
                AgentConfig,
                ConfigLoader,
            )
            from utu.utils import PrintUtils, setup_logging  # type: ignore # noqa: F401

            self._initialized = True
        except ImportError as e:
            logger.error(f"utu modules import failed: {e}")
            logger.error(f"Working directory: {os.getcwd()}")
            logger.error(f"utu path exists: {(agent_dir / 'utu').exists()}")
            raise ExecutionError(f"Failed to import utu modules: {e}")

    def _setup_environment_from_secrets(self) -> None:
        """Setup environment variables from secrets.yaml if not already set"""
        import yaml

        secrets_path = agent_dir / "secrets.yaml"

        if not secrets_path.exists():
            logger.warning(f"secrets.yaml not found at {secrets_path}")
            return

        try:
            with open(secrets_path, encoding="utf-8") as f:
                secrets = yaml.safe_load(f) or {}

            # Set UTU environment variables if not already set
            # Read configuration values directly from secrets.yaml
            env_mappings = {
                "UTU_LLM_TYPE": secrets.get("UTU_LLM_TYPE"),
                "UTU_LLM_MODEL": secrets.get("UTU_LLM_MODEL"),
                "UTU_LLM_BASE_URL": secrets.get("UTU_LLM_BASE_URL"),
                "UTU_LLM_API_KEY": secrets.get("UTU_LLM_API_KEY"),
                "SERPER_API_KEY": secrets.get("SERPER_API_KEY"),
                "JINA_API_KEY": secrets.get("JINA_API_KEY"),
                "DB_URL": secrets.get("DB_URL"),
            }

            # Record successfully loaded configurations
            loaded_configs = []
            for env_var, value in env_mappings.items():
                if value and not os.getenv(env_var):
                    os.environ[env_var] = str(value)
                    if "API_KEY" in env_var:
                        # Only show first few and last few characters of API keys
                        masked_value = f"{str(value)[:6]}...{str(value)[-4:]}"
                        logger.info(
                            f"✅ Loaded from secrets.yaml: {env_var} = {masked_value}"
                        )
                    else:
                        logger.info(f"✅ Loaded from secrets.yaml: {env_var} = {value}")
                    loaded_configs.append(env_var)
                elif os.getenv(env_var):
                    logger.debug(f"Environment variable already exists: {env_var}")
                else:
                    logger.warning(f"⚠️ Missing configuration: {env_var}")

            if loaded_configs:
                logger.info(
                    f"📋 Loaded {len(loaded_configs)} configuration items in total"
                )
            else:
                logger.warning("⚠️ No configurations loaded from secrets.yaml")

        except Exception as e:
            logger.warning(f"Failed to load secrets.yaml: {e}")
            # Continue without secrets, will use environment variables if available

    def prepare_data(self, spec: AgentSpecStrict) -> None:
        """Prepare task list from various data sources"""
        data = spec.data or {}

        # Backward compatibility: if no data config, use spec.task
        if not data:
            task_input = spec.task or ""
            if not task_input:
                raise ExecutionError("Either spec.data or spec.task is required")
            self._tasks = [task_input]
            return

        dtype = data.get("type")

        if dtype == "static":
            task_input = data.get("task", "")
            if not task_input:
                raise ExecutionError("spec.data.task is required for type == 'static'")
            self._tasks = [task_input]

        elif dtype == "dataset":
            data_url = data.get("url")
            if not data_url:
                raise ExecutionError("spec.data.url is required for type == 'dataset'")

            name = data.get("name", None)
            split = data.get("split", DEFAULT_DATASET_SPLIT)
            shuffle = bool(data.get("shuffle", False))

            trust_remote_code = data.get("trust_remote_code")
            revision = data.get("revision")
            dataset_kwargs = {
                "name": name,
                "split": split,
                "revision": revision,
            }
            if trust_remote_code is not None:
                dataset_kwargs["trust_remote_code"] = bool(trust_remote_code)
            dataset = load_dataset(
                data_url,
                **{k: v for k, v in dataset_kwargs.items() if v is not None},
            )

            if shuffle:
                seed = int(data.get("seed", DEFAULT_SHUFFLE_SEED))
                buffer_size = data.get("buffer_size", None)
                if buffer_size is None:
                    dataset = dataset.shuffle(seed=seed)
                else:
                    dataset = dataset.shuffle(
                        seed=seed,
                        buffer_size=int(buffer_size),  # type: ignore
                    )

            column = data.get("column", DEFAULT_DATASET_COLUMN)
            if column not in dataset.column_names:
                raise ExecutionError(
                    f"Column '{column}' not found in dataset. "
                    f"Available: {dataset.column_names}"
                )
            self._tasks = [str(x) for x in dataset[column]]

        elif dtype == "list":
            items = data.get("items", [])
            if not isinstance(items, list) or any(
                not isinstance(x, str) for x in items
            ):
                raise ExecutionError(
                    "spec.data.items must be a list of strings for type == 'list'"
                )
            self._tasks = items

        elif dtype == "graph_template":
            prompts = build_prompts_from_graph_template(data, spec)
            self._tasks = prompts  # type: ignore

        else:
            raise ExecutionError(f"Unsupported spec.data.type: {dtype!r}")

    def run(self, task: ExecutorTask, out_dir: Path) -> AgentResult:
        """Execute agent tasks using youtu-agent (utu) framework"""
        self.ensure_dir(out_dir)

        # Initialize utu modules
        self.prepare()

        spec = self.require_spec(task, AgentSpecStrict)
        agent_config_name = spec.configName or DEFAULT_CONFIG_NAME
        task_timeout = _resolve_task_timeout(spec.agent)

        # Prepare tasks from various data sources
        self.prepare_data(spec)

        if not self._tasks:
            raise ExecutionError("No tasks prepared. Check spec.data configuration")

        try:
            if len(self._tasks) == 1:
                # Single task execution (backward compatible)
                result = asyncio.run(
                    self._run_agent_task(
                        agent_config_name, self._tasks[0], out_dir, task_timeout
                    )
                )
                agent_output_ref = result.get("_agent_output_ref")

                output = AgentResult(
                    model=agent_config_name,
                    items=[
                        {
                            "index": 0,
                            "output": result.get("output", ""),
                            "finish_reason": "completed",
                        }
                    ],
                    usage={
                        "execution_time_sec": result.get("usage", {}).get(
                            "execution_time_sec", 0
                        ),
                        "num_requests": 1,
                        "agent_config": agent_config_name,
                    },
                    metadata={
                        "task": self._tasks[0],
                        "execution_log": result.get("log", []),
                    },
                    agent_output=(
                        ArtifactRef(path=agent_output_ref)
                        if isinstance(agent_output_ref, str)
                        else None
                    ),
                )
            else:
                # Batch execution for multiple tasks
                results = asyncio.run(
                    self._run_agent_tasks_batch(
                        agent_config_name, self._tasks, out_dir, task_timeout
                    )
                )

                # Transform batch results to standard format
                items = []
                for item in results.get("items", []):
                    items.append(
                        {
                            "index": item["index"],
                            "output": item["response"],
                            "finish_reason": (
                                "completed"
                                if item["status"] == "completed"
                                else "failed"
                            ),
                        }
                    )

                batch_summary_ref = results.get("_batch_summary_ref")
                output = AgentResult(
                    model=agent_config_name,
                    items=items,
                    usage={
                        "execution_time_sec": results.get("usage", {}).get(
                            "execution_time_sec", 0
                        ),
                        "num_requests": len(self._tasks),
                        "agent_config": agent_config_name,
                    },
                    metadata={
                        "tasks_count": len(self._tasks),
                        "execution_log": results.get("log", []),
                        "batch_summary": results.get("batch_summary", {}),
                    },
                    batch_summary_file=(
                        ArtifactRef(path=batch_summary_ref)
                        if isinstance(batch_summary_ref, str)
                        else None
                    ),
                )

            maybe_upload_artifacts(task, out_dir, logger=logger)

            logger.info(f"Agent execution completed: {len(self._tasks)} task(s)")
            return output

        except ExecutionError:
            raise
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            error_output = AgentResult(
                ok=False,
                model=agent_config_name,
                items=[],
                usage={
                    "execution_time_sec": 0,
                    "num_requests": len(self._tasks),
                    "agent_config": agent_config_name,
                },
                metadata={
                    "tasks_count": len(self._tasks),
                    "error": str(e),
                    "execution_log": [],
                },
            )
            write_executor_result(
                out_dir / "results.json", task.task_id, task.spec, error_output
            )
            raise ExecutionError(f"Agent execution failed: {e}") from e

    async def _run_agent_task(
        self,
        config_name: str,
        task_input: str,
        out_dir: Path,
        task_timeout: int,
    ) -> dict[str, Any]:
        """Execute a single agent task end-to-end via the utu framework.

        Streams progress events, enforces ``task_timeout`` on the streaming
        run, and writes the final agent output to ``artifacts/agent_output.txt``.
        """
        import asyncio

        from utu.agents import get_agent  # type: ignore # noqa: F401
        from utu.config import AgentConfig, ConfigLoader  # type: ignore # noqa: F401
        from utu.tracing.setup import setup_tracing  # type: ignore # noqa: F401

        logger.info("🚀 Starting agent task execution")
        logger.info(f"📋 Configuration: {config_name}")
        logger.info(f"📝 Task description: {task_input[:200]}...")
        logger.debug(f"Output directory: {out_dir}")

        # Setup tracing (optional)
        try:
            setup_tracing()
            logger.debug("Tracing setup completed")
        except Exception as e:
            logger.warning(f"Failed to setup tracing: {e}")

        # Setup utu logging directory
        utu_log_dir = out_dir / "artifacts" / "logs" / "utu"
        utu_log_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created log directory: {utu_log_dir}")

        execution_log = []
        usage_stats: dict[str, Any] = {}

        try:
            # Load agent configuration
            logger.info("Loading agent configuration...")
            try:
                config: AgentConfig = ConfigLoader.load_agent_config(config_name)
                logger.info(
                    "✅ Configuration loaded: "
                    f"{config.name if hasattr(config, 'name') else config_name}"
                )
                execution_log.append(f"Loaded config: {config_name}")
            except Exception as config_error:
                logger.error(
                    f"Failed to load agent config '{config_name}': {config_error}"
                )
                logger.error(f"Config error type: {type(config_error).__name__}")
                raise ExecutionError(
                    f"Failed to load agent config '{config_name}': {config_error}"
                )

            # Create agent instance
            logger.info("Creating agent instance...")
            agent = get_agent(config)
            logger.info(f"✅ Agent created: {type(agent).__name__}")
            execution_log.append(f"Created agent: {type(agent).__name__}")

            # Build agent (loading tools and models)
            logger.info("Building agent (loading tools and models)...")
            await asyncio.wait_for(agent.build(), timeout=120)  # 2 minute timeout
            logger.info("✅ Agent built successfully")
            execution_log.append("Agent built successfully")

            # Execute the task
            logger.info("🎯 Starting task execution...")
            logger.debug(f"Task input length: {len(task_input)} characters")
            execution_log.append(f"Starting task execution: {task_input[:100]}...")

            # Log task timeout
            logger.debug(f"Task timeout set to {task_timeout} seconds")

            # Use streaming execution (agent supports it based on usage patterns)
            logger.info("Using streaming execution with progress tracking...")
            result_streaming = agent.run_streamed(input=task_input)

            # ``asyncio.timeout`` wraps the entire streaming run so the
            # deadline applies even when the agent stalls before emitting an
            # event — a bare ``async for`` would otherwise block forever.
            async with asyncio.timeout(task_timeout):
                # Monitor execution stream
                async for event in result_streaming.stream_events():
                    # Check agent switching events
                    if hasattr(event, "new_agent") and event.new_agent:
                        agent_name = getattr(event.new_agent, "name", "Unknown")
                        logger.info(f"🤖 Agent switched: {agent_name}")
                        execution_log.append(f"Agent switched: {agent_name}")

                    # Check Orchestra task planning events
                    elif hasattr(event, "name") and hasattr(event, "item"):
                        if event.name == "plan":
                            logger.info("📋 Task planning completed")
                            execution_log.append("Task planning completed")
                            if event.item:
                                plan_summary = str(event.item)[:200]
                                logger.info(f"📝 Planning content: {plan_summary}...")

                    # Check RunItem events (tool calls, outputs, etc.)
                    elif hasattr(event, "item") and hasattr(event.item, "type"):
                        item = event.item
                        item_type = item.type
                        agent_name = getattr(
                            getattr(item, "agent", None), "name", "Unknown"
                        )

                        if item_type == "tool_call_item":
                            if hasattr(item, "raw_item"):
                                tool_name = getattr(
                                    item.raw_item, "name", "Unknown Tool"
                                )
                                tool_args = getattr(item.raw_item, "arguments", "")[
                                    :100
                                ]
                                logger.info(
                                    f"🔧 [{agent_name}] "
                                    f"Tool call: {tool_name}({tool_args}...)"
                                )
                                execution_log.append(f"Tool call: {tool_name}")

                        elif item_type == "tool_call_output_item":
                            if hasattr(item, "output"):
                                output_preview = str(item.output)[:150]
                                logger.info(
                                    f"📤 [{agent_name}] "
                                    f"Tool output: {output_preview}..."
                                )
                                execution_log.append("Tool output received")

                        elif item_type == "message_output_item":
                            if hasattr(item, "content") and item.content:
                                for content_item in item.content:
                                    if getattr(content_item, "type", None) == "text":
                                        text_preview = getattr(
                                            content_item, "text", ""
                                        )[:100]
                                        if text_preview.strip():
                                            logger.info(
                                                f"💬 [{agent_name}] "
                                                f"Output: {text_preview}..."
                                            )

                        elif item_type == "reasoning_item":
                            if hasattr(item, "summary"):
                                for summary_item in item.summary:
                                    if (
                                        getattr(summary_item, "type", None)
                                        == "summary_text"
                                    ):
                                        reasoning_text = getattr(
                                            summary_item, "text", ""
                                        )[:100]
                                        if reasoning_text.strip():
                                            logger.info(
                                                f"🧠 [{agent_name}] "
                                                f"Reasoning: {reasoning_text}..."
                                            )

                    # Check raw response events (including token generation)
                    elif hasattr(event, "data") and hasattr(event.data, "type"):
                        # Character counting is removed.
                        pass

                # Wait for streaming result to complete
                await result_streaming._run_impl_task
            result = result_streaming
            logger.info("✅ Execution completed")
            execution_log.append("Streaming execution completed")

        except TimeoutError:
            logger.error(f"⏰ Task timeout ({task_timeout} seconds)")
            raise ExecutionError("Agent task timed out")
        except Exception as e:
            logger.error(f"❌ Error during execution: {str(e)}")
            raise ExecutionError(f"Agent task failed: {str(e)}")

        # Extract output based on result type
        if hasattr(result, "final_output") and result.final_output:
            output = str(result.final_output)
            logger.info(f"📄 Final output extracted: {len(output)} characters")
        else:
            # Fallback: try common attributes
            output = (
                getattr(result, "output", None)
                or getattr(result, "final", None)
                or str(result)
            )
            logger.info(f"📄 Output extracted (fallback): {len(output)} characters")

        execution_log.append(f"Generated output: {len(output)} characters")

        # Save output to file
        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_file = artifacts_dir / "agent_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"💾 Output saved to: {output_file}")
        execution_log.append("Saved output to file")

        # Log first 100 characters of output for debugging
        logger.info(
            f"Final output preview ({len(output)} characters): {output[:100]}..."
        )

        # Cleanup agent resources
        try:
            logger.debug("Starting agent cleanup...")
            await asyncio.wait_for(agent.cleanup(), timeout=30)
            logger.debug("Agent cleanup completed")
            execution_log.append("Agent cleanup completed")
        except Exception as e:
            logger.warning(f"Agent cleanup failed: {e}")
            execution_log.append(f"Agent cleanup failed: {e}")

        return {
            "output": output,
            "log": execution_log,
            "usage": usage_stats,
            "_agent_output_ref": "agent_output.txt",
        }

    async def _run_agent_tasks_batch(
        self,
        config_name: str,
        tasks: list[str],
        out_dir: Path,
        task_timeout: int,
    ) -> dict[str, Any]:
        """Execute multiple agent tasks in batch under a shared agent build.

        Each task runs under the same ``task_timeout`` as the single-task
        path; failures are recorded per task without aborting the batch.
        """
        from utu.agents import get_agent  # type: ignore # noqa: F401
        from utu.config import AgentConfig, ConfigLoader  # type: ignore # noqa: F401
        from utu.tracing.setup import setup_tracing  # type: ignore # noqa: F401

        logger.info(f"Starting batch execution of {len(tasks)} tasks")

        # Setup tracing
        try:
            setup_tracing()
        except Exception as e:
            logger.warning(f"Failed to setup tracing: {e}")

        # Setup log directory
        batch_log_dir = out_dir / "artifacts" / "logs" / "batch"
        batch_log_dir.mkdir(parents=True, exist_ok=True)

        execution_log = []
        all_items = []
        usage_stats: dict[str, Any] = {}

        try:
            # Load agent configuration once
            config: AgentConfig = ConfigLoader.load_agent_config(config_name)
            agent = get_agent(config)

            # Build agent once for all tasks
            await asyncio.wait_for(agent.build(), timeout=120)
            execution_log.append("Agent built for batch processing")

            for i, task_input in enumerate(tasks):
                logger.info(f"Processing task {i+1}/{len(tasks)}")
                execution_log.append(f"Starting task {i+1}: {task_input[:50]}...")

                try:
                    # Use streaming execution
                    result_streaming = agent.run_streamed(input=task_input)

                    # ``asyncio.timeout`` covers both the event loop and the
                    # impl-task wait so a stall before any event is still
                    # bounded by ``task_timeout``.
                    async with asyncio.timeout(task_timeout):
                        # Process stream events (simplified for batch)
                        async for event in result_streaming.stream_events():
                            # Log major events only to avoid overwhelming logs
                            if hasattr(event, "new_agent") and event.new_agent:
                                agent_name = getattr(event.new_agent, "name", "Unknown")
                                logger.debug(
                                    f"Task {i+1}: Agent switched to {agent_name}"
                                )

                        # Wait for completion (still under timeout).
                        await result_streaming._run_impl_task
                    result = result_streaming

                    # Extract output
                    if hasattr(result, "final_output") and result.final_output:
                        output = str(result.final_output)
                    else:
                        output = getattr(result, "output", None) or str(result)

                    # Save individual task output
                    task_output_file = batch_log_dir / f"task_{i+1}_output.txt"
                    with open(task_output_file, "w", encoding="utf-8") as f:
                        f.write(output)

                    all_items.append(
                        {
                            "index": i,
                            "task": task_input,
                            "response": output,
                            "status": "completed",
                        }
                    )

                    execution_log.append(f"Task {i+1} completed: {len(output)} chars")

                except Exception as e:
                    logger.error(f"Task {i+1} failed: {e}")
                    all_items.append(
                        {
                            "index": i,
                            "task": task_input,
                            "response": "",
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    execution_log.append(f"Task {i+1} failed: {e}")

            # Cleanup agent
            try:
                await asyncio.wait_for(agent.cleanup(), timeout=30)
                execution_log.append("Batch agent cleanup completed")
            except Exception as e:
                logger.warning(f"Batch agent cleanup failed: {e}")

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise ExecutionError(f"Batch execution failed: {e}")

        # Save batch summary
        batch_summary = {
            "total_tasks": len(tasks),
            "completed": len(
                [item for item in all_items if item["status"] == "completed"]
            ),
            "failed": len([item for item in all_items if item["status"] == "failed"]),
        }

        artifacts_dir = out_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_file = artifacts_dir / "batch_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)

        return {
            "items": all_items,
            "log": execution_log,
            "usage": usage_stats,
            "batch_summary": batch_summary,
            "_batch_summary_ref": "batch_summary.json",
        }
