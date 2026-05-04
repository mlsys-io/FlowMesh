"""SSH session commands."""

import asyncio
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import typer
import websockets
from flowmesh import FlowMesh
from flowmesh.exceptions import FlowMeshError, NotFoundError
from flowmesh.models.common import TERMINAL_TASK_STATUSES, TaskStatus
from flowmesh.params import append_param, extend_params

from ..core import logging
from ..core.query import parse_query_filters
from ..core.typer import get_typer

app = get_typer(help="Connect to SSH sessions on FlowMesh workers.")


def _exec_ssh(
    ssh_info: dict[str, Any],
    task_id: str,
    extra_args: str | None,
    direct: bool = False,
) -> None:
    """Replace the current process with an ssh command."""
    ssh_bin = shutil.which("ssh")
    if ssh_bin is None:
        logging.error("ssh not found in PATH")
        raise typer.Exit(code=1)

    mode = str(ssh_info.get("mode", "direct"))
    user = ssh_info.get("username", "flowmesh")
    host = ssh_info.get("host")
    port = ssh_info.get("port")
    if direct:
        if direct_host := ssh_info.get("directHost"):
            host = direct_host
        if direct_port := ssh_info.get("directPort"):
            port = direct_port

    args = [
        ssh_bin,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
    ]

    if direct or mode in ("direct", "forward"):
        if not host:
            logging.error("SSH host is not available.")
            raise typer.Exit(code=1)
        if port:
            args += ["-p", str(port)]
        args.append(f"{user}@{host}")
    elif mode == "proxy":
        proxy_cmd = f"flowmesh ssh proxy {task_id}"
        args += ["-o", f"ProxyCommand={proxy_cmd}"]
        args.append(f"{user}@{task_id}")
    else:
        logging.error(f"Unknown SSH publish mode: {mode}")
        raise typer.Exit(code=1)

    if extra_args:
        args.extend(extra_args.split())

    logging.info(f"Connecting: {' '.join(args)}")
    os.execvp(ssh_bin, args)


def _exit_for_task_status(task_id: str, task_info: Any) -> None:
    status = task_info.status
    if status == TaskStatus.DONE:
        logging.info(f"Task {task_id} completed successfully.")
        raise typer.Exit(code=0)
    if status == TaskStatus.FAILED:
        logging.error(f"Task {task_id} failed: {task_info.error or 'unknown error'}")
        raise typer.Exit(code=1)
    if status == TaskStatus.CANCELLED:
        logging.info(f"Task {task_id} was cancelled.")
        raise typer.Exit(code=130)


def _poll_status_and_exit(client: FlowMesh, task_id: str, interval: float) -> None:
    try:
        task_info = client.tasks.wait(task_id, interval=interval)
    except KeyboardInterrupt:
        logging.warning("Interrupted.")
        raise typer.Exit(code=130)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    _exit_for_task_status(task_id, task_info)


def _stream_logs_and_exit(
    client: FlowMesh,
    task_id: str,
    interval: float,
    tail: bool = False,
) -> None:
    logging.info("Streaming task logs...")
    cursor: str | None = "$" if tail else "0"
    try:
        while True:
            saw_logs = False
            try:
                for entry in client.tasks.stream_logs(task_id, cursor=cursor):
                    saw_logs = True
                    cursor = entry.cursor or cursor
                    event = entry.event.model_dump(mode="json")
                    ts = event.get("ts", "")
                    message = str(event.get("message", "")).rstrip("\n")
                    prefix = f"[{ts}] " if ts else ""
                    logging.log(prefix + message)
            except NotFoundError:
                pass

            task_info = client.tasks.retrieve(task_id)
            if task_info.status in TERMINAL_TASK_STATUSES:
                _exit_for_task_status(task_id, task_info)
            if not saw_logs:
                time.sleep(interval)
    except KeyboardInterrupt:
        logging.warning("Interrupted.")
        raise typer.Exit(code=130)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)


# ------------------------------------------------------------------ #
# Commands
# ------------------------------------------------------------------ #


@app.command("connect")
def connect(
    task_id: str = typer.Argument(..., help="Task identifier"),
    interval: float = typer.Option(2.0, help="Polling interval in seconds"),
    direct: bool = typer.Option(
        False, "--direct", help="Use direct SSH instead of proxy/forward"
    ),
    ssh_args: str | None = typer.Option(
        None, "--ssh-args", help="Extra arguments passed to ssh"
    ),
) -> None:
    """Wait for an SSH session to be ready on a task and connect."""
    logging.info(f"Waiting for SSH session on task {task_id}...")
    client = FlowMesh()
    try:
        ssh_info = client.tasks.wait_for_ssh(task_id, interval=interval)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    _exec_ssh(ssh_info, task_id, ssh_args, direct)


@app.command("run")
def run(
    key: Path | None = typer.Option(
        None, "--key", "-k", help="Path to SSH public key file"
    ),
    gpu: int | None = typer.Option(None, "--gpu", "-g", help="Number of GPUs"),
    gpu_memory: str | None = typer.Option(
        None, "--gpu-memory", help="GPU memory requirement (e.g. 16GB)"
    ),
    memory: str | None = typer.Option(
        None, "--memory", "-m", help="System memory (e.g. 8Gi)"
    ),
    cpu: int | None = typer.Option(None, "--cpu", help="Number of CPU cores"),
    ttl: int = typer.Option(3600, "--ttl", help="Session TTL in seconds"),
    idle_timeout: int = typer.Option(
        900, "--idle-timeout", help="Idle timeout in seconds"
    ),
    image: str | None = typer.Option(None, "--image", help="Custom session image"),
    user: str = typer.Option("flowmesh", "--user", "-u", help="SSH username"),
    mode: str = typer.Option(
        "proxy", "--mode", help="Publish mode: direct|proxy|forward"
    ),
    worker: str | None = typer.Option(
        None, "--worker", "-w", help="Pin to a specific worker"
    ),
    name: str = typer.Option("ssh-session", "--name", "-n", help="Task name"),
    env: list[str] | None = typer.Option(
        None, "--env", "-e", help="Environment variable KEY=VALUE (repeatable)"
    ),
    command: str | None = typer.Option(
        None,
        "--command",
        "-c",
        help="Command to run non-interactively (e.g. 'python train.py')",
    ),
    entrypoint_override: str | None = typer.Option(
        None,
        "--entrypoint",
        help="Override image entrypoint (non-interactive mode)",
    ),
    interactive: bool | None = typer.Option(
        None,
        "--interactive/--non-interactive",
        help="Whether to use interactive SSH or non-interactive command mode.",
    ),
    logs: bool = typer.Option(
        False,
        "--logs/--no-logs",
        help="For non-interactive runs, stream logs instead of only polling status.",
    ),
    tail: bool = typer.Option(
        False,
        "--tail",
        help="With --logs, start from the latest log entry instead of the beginning.",
    ),
    interval: float = typer.Option(2.0, "--interval", help="Polling interval"),
    direct: bool = typer.Option(
        False, "--direct", help="Use direct SSH instead of proxy/forward"
    ),
    ssh_args: str | None = typer.Option(
        None, "--ssh-args", help="Extra arguments passed to ssh"
    ),
) -> None:
    """Submit an SSH task, wait for the session, and connect.

    When --command or --entrypoint is provided, the task runs non-interactively:
    the specified command executes in the container and the CLI polls until the
    task completes.
    """
    inferred_interactive = command is None and entrypoint_override is None
    if interactive and not inferred_interactive:
        logging.error(
            "Interactive SSH mode cannot be combined with --command or --entrypoint."
        )
        raise typer.Exit(code=1)
    resolved_interactive = (
        interactive if interactive is not None else inferred_interactive
    )
    noninteractive = not resolved_interactive
    cmd_list: list[str] | None = shlex.split(command) if command is not None else None
    ep_list: list[str] | None = (
        shlex.split(entrypoint_override) if entrypoint_override is not None else None
    )

    public_key: str | None = None
    if resolved_interactive:
        if key:
            if not key.exists():
                logging.error(f"Public key not found: {key}")
                raise typer.Exit(code=1)
            public_key = key.read_text().strip()
        else:
            try:
                public_key = FlowMesh().ssh.detect_public_key()
            except FlowMeshError as exc:
                logging.error(str(exc))
                raise typer.Exit(code=1)

        if mode not in ("direct", "proxy", "forward"):
            logging.error(f"Invalid mode: {mode}. Use 'direct', 'proxy', or 'forward'.")
            raise typer.Exit(code=1)

    client = FlowMesh()
    workflow_yaml = client.ssh.build_task_yaml(
        name=name,
        public_key=public_key,
        user=user,
        mode=mode,
        ttl=ttl,
        idle_timeout=idle_timeout,
        gpu=gpu,
        gpu_memory=gpu_memory,
        cpu=cpu,
        memory=memory,
        image=image,
        worker=worker,
        env_pairs=env,
        interactive=resolved_interactive,
        command=cmd_list,
        entrypoint=ep_list,
    )

    logging.info("Submitting SSH task...")
    try:
        result = client.workflows.submit(workflow_yaml)
    except FlowMeshError as exc:
        logging.error(f"Failed to submit SSH task: {exc}")
        raise typer.Exit(code=1)

    task_id = None
    if result.tasks:
        task_id = result.tasks[0].task_id
    if not task_id:
        logging.error("No task ID returned from submission.")
        logging.error(result.model_dump_json(indent=2))
        raise typer.Exit(code=1)

    logging.info(f"Task submitted: {task_id}")

    if noninteractive:
        if logs:
            _stream_logs_and_exit(client, task_id, interval, tail)
        _poll_status_and_exit(client, task_id, interval)
    else:
        # Interactive: wait for SSH session and connect.
        logging.info("Waiting for SSH session...")
        try:
            ssh_info = client.tasks.wait_for_ssh(task_id, interval=interval)
        except FlowMeshError as exc:
            logging.error(str(exc))
            raise typer.Exit(code=1)
        _exec_ssh(ssh_info, task_id, ssh_args, direct)


@app.command("proxy")
def proxy(
    task_id: str = typer.Argument(..., help="Task identifier"),
) -> None:
    """Raw stdin/stdout proxy for SSH ProxyCommand (internal)."""
    try:
        asyncio.run(_run_proxy(task_id))
    except KeyboardInterrupt:
        pass


async def _run_proxy(task_id: str) -> None:
    """Async WebSocket <-> stdio proxy."""
    client = FlowMesh()
    api_key = client.api_key
    ws_url = client.ssh.proxy_url(task_id)

    auth_header = {"Authorization": f"Bearer {api_key}"} if api_key else None
    async with websockets.connect(ws_url, additional_headers=auth_header) as ws:
        loop = asyncio.get_running_loop()

        async def stdin_to_ws() -> None:
            try:
                while True:
                    data = await loop.run_in_executor(
                        None, os.read, sys.stdin.fileno(), 4096
                    )
                    if not data:
                        break
                    await ws.send(data)
            except Exception:
                pass

        async def ws_to_stdout() -> None:
            try:
                async for msg in ws:
                    if isinstance(msg, str):
                        msg = msg.encode()
                    await loop.run_in_executor(None, os.write, sys.stdout.fileno(), msg)
            except Exception:
                pass

        t1 = asyncio.create_task(stdin_to_ws())
        t2 = asyncio.create_task(ws_to_stdout())
        _, pending = await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


@app.command("connections")
def list_connections(
    connection_id: str | None = typer.Option(
        None, "--id", help="Filter by connection id"
    ),
    access_mode: list[str] | None = typer.Option(
        None, "--mode", help="Filter by access mode (repeatable)"
    ),
    task_id: str | None = typer.Option(None, "--task-id", help="Filter by task id"),
    workflow_id: str | None = typer.Option(
        None, "--workflow-id", help="Filter by workflow id"
    ),
    worker_id: str | None = typer.Option(
        None, "--worker-id", help="Filter by worker id"
    ),
    node_id: str | None = typer.Option(None, "--node-id", help="Filter by node id"),
    username: str | None = typer.Option(
        None, "--username", help="Filter by SSH username"
    ),
    source_ip: str | None = typer.Option(
        None, "--source-ip", help="Filter by source IP"
    ),
    source_port: int | None = typer.Option(
        None, "--source-port", help="Filter by source port"
    ),
    query: list[str] | None = typer.Option(
        None, "--query", "-q", help="Filter connections by key=value pairs"
    ),
) -> None:
    """List active SSH connections audited by the server."""
    client = FlowMesh()
    query_params = parse_query_filters(query)
    append_param(query_params, "connection_id", connection_id)
    extend_params(query_params, "access_mode", access_mode)
    append_param(query_params, "task_id", task_id)
    append_param(query_params, "workflow_id", workflow_id)
    append_param(query_params, "worker_id", worker_id)
    append_param(query_params, "node_id", node_id)
    append_param(query_params, "username", username)
    append_param(query_params, "source_ip", source_ip)
    append_param(query_params, "source_port", source_port)
    try:
        connections = client.ssh.list(query_params=query_params)
    except FlowMeshError as exc:
        logging.error(str(exc))
        raise typer.Exit(code=1)
    logging.log(json.dumps([c.model_dump(mode="json") for c in connections], indent=2))
