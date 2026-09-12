"""Process session backend: sshd as a process beside the worker.

For deployments where the worker container *is* the machine the user rents —
vast.ai instances have no Docker socket — and where the supervisor that dials
the relay uplink lives somewhere else entirely, so loopback is not a reachable
relay target.

Three consequences follow from having no container around the session and are
enforced here rather than assumed:

* **One session per worker.** Sessions sharing a worker would share its
  filesystem and process namespace, so a second concurrent session is refused.
* **No worker-side resource cap.** ``SSH_MAX_CPU`` / ``SSH_MAX_MEMORY`` /
  ``SSH_MAX_PIDS`` need cgroup control the worker does not have over itself;
  the size of the rented box is the cap.
* **Isolation depends on the worker's own privileges.** A root worker gives
  each session its own account, so the session cannot reach the worker's
  credentials. A non-root worker can only authenticate its own account, and
  the session inherits everything that account can read.
"""

import ctypes
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from shared.tasks.worker_message import WorkerHardware
from shared.utils import parse_float_env
from shared.utils.manifest import ARTIFACTS_DIR
from worker.config import WorkerConfig

from ..base_executor import ExecutionError
from .base import (
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
    count_established_connections,
    is_ssh_ready,
    path_size_bytes,
    read_local_proc_net_tcp,
    resolve_tailnet_address,
)
from .config import (
    STOP_TIMEOUT_SEC,
    SSHConfig,
    normalize_mount_path,
    reserve_mount_path,
)
from .inputs import stage_inputs_locally
from .session_identity import SessionIdentity, reap_stale_accounts, resolve_identity

logger = logging.getLogger(__name__)

_SSHD_CANDIDATES = ("/usr/sbin/sshd", "/usr/local/sbin/sshd", "sshd")
_KEYGEN_BINARY = "ssh-keygen"
_KEYGEN_TIMEOUT_SEC = 30.0
_TERMINATE_GRACE_SEC = 5.0
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VALUE_FORBIDDEN = ('"', "\\", "\n", "\r")
_PORT_ATTEMPTS = 3
_READY_PROBE_SEC = 2.0
_BIND_FAILURE_MARKERS = ("cannot bind", "address already in use", "bind to port")
_PR_SET_DUMPABLE = 4
_SPAWN_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ")
# sshd's own default; the session's PATH prepends the per-session bin dir.
_DEFAULT_SESSION_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/games"


def find_sshd() -> str | None:
    for candidate in _SSHD_CANDIDATES:
        if candidate.startswith("/"):
            if os.access(candidate, os.X_OK):
                return candidate
        elif resolved := shutil.which(candidate):
            return resolved
    return None


def find_ssh_keygen() -> str | None:
    return shutil.which(_KEYGEN_BINARY)


class ProcessSessionBackend(SSHSessionBackend):
    name = "process"
    supports_noninteractive = False

    def __init__(
        self, config: WorkerConfig, hardware: WorkerHardware | None = None
    ) -> None:
        super().__init__(config, hardware)
        self._lock = threading.Lock()
        self._active: ProcessSession | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        if find_sshd() is None:
            logger.info(
                "Process SSH backend unavailable: no sshd binary found "
                "(install openssh-server in the worker image)"
            )
            return False
        if find_ssh_keygen() is None:
            logger.info(
                "Process SSH backend unavailable: no %s binary found", _KEYGEN_BINARY
            )
            return False
        return True

    def prepare(self) -> None:
        reap_stale_accounts(keep=self._active_account_name())
        if self._config.ssh_limits is not None:
            logger.warning(
                "SSH resource caps are configured but the process backend cannot "
                "enforce them; the size of this worker is the cap"
            )
        if self._config.enable_ssh_gpu_limit:
            logger.warning(
                "ENABLE_SSH_GPU_LIMIT is set, but the process backend can only hand "
                "the session a CUDA_VISIBLE_DEVICES value it is free to unset; the "
                "GPU subset is advisory here, not enforced"
            )
        if os.getuid() != 0 and not _hide_worker_environ():
            logger.warning(
                "Could not restrict access to this worker's process environment"
            )

    def _default_relay_host(self) -> str:
        address = resolve_tailnet_address()
        if address is None:
            raise ExecutionError(
                "Cannot publish a relay target for this SSH session: the worker has "
                "no tailnet address and SSH_RELAY_HOST is unset. A proxy- or "
                "forward-mode session needs an address the supervisor can dial, and "
                "loopback is only correct when the supervisor shares this host."
            )
        return address

    def session_host(self) -> str:
        if override := self._config.ssh_relay_host:
            return override
        return resolve_tailnet_address() or socket.getfqdn()

    def start_session(self, request: SessionRequest) -> "ProcessSession":
        cfg = request.cfg
        if not cfg.interactive:
            raise ExecutionError(
                "Non-interactive SSH tasks need a container runtime to run the "
                "requested image; this worker runs SSH sessions as processes. "
                "Route the task to a worker with a Docker daemon."
            )
        with self._lock:
            if self._active is not None:
                raise ExecutionError(
                    "This worker already has an SSH session; process-mode sessions "
                    "share a filesystem and process namespace, so isolation is by "
                    "worker and only one session may run at a time."
                )
            session = self._create_session(request)
            self._active = session
        return session

    def teardown(self, worker_name: str) -> None:
        with self._lock:
            session = self._active
        if session is None:
            return
        session.stop(parse_float_env("SSH_STOP_TIMEOUT_SEC", STOP_TIMEOUT_SEC))
        session.cleanup()

    def _release(self, session: "ProcessSession") -> None:
        with self._lock:
            if self._active is session:
                self._active = None

    def _active_account_name(self) -> str | None:
        with self._lock:
            return self._active.identity.name if self._active else None

    # ------------------------------------------------------------------ #
    # Session construction
    # ------------------------------------------------------------------ #

    def _create_session(self, request: SessionRequest) -> "ProcessSession":
        cfg = request.cfg
        sshd_path = find_sshd()
        keygen_path = find_ssh_keygen()
        if sshd_path is None or keygen_path is None:
            raise ExecutionError(
                "SSH session cannot start: sshd or ssh-keygen is missing from this "
                "worker image"
            )
        if cfg.image:
            logger.info(
                "Ignoring SSH spec image %s: process-mode sessions run in the "
                "worker's own root filesystem",
                cfg.image,
            )

        session_dir = Path(
            tempfile.mkdtemp(prefix=f"flowmesh-ssh-{request.session_id[:8]}-")
        )
        session_dir.chmod(0o700)
        identity: SessionIdentity | None = None
        plan: ProcessSessionPaths | None = None
        try:
            identity = resolve_identity(request.session_id, session_dir)
            if identity.isolates_from_worker:
                # The session traverses into its home without being able to list
                # the host key and sshd_config sitting beside it.
                session_dir.chmod(0o711)
                if cfg.user != identity.name:
                    logger.info(
                        "Ignoring SSH spec user %s: this session logs in as its own "
                        "account %s",
                        cfg.user,
                        identity.name,
                    )
            plan = self._build_paths(request, session_dir, identity)
            host_key = session_dir / "ssh_host_ed25519_key"
            _generate_host_key(keygen_path, host_key)
            environment = self._build_environment(
                cfg.user,
                cfg.authorized_keys,
                cfg.extra_env,
                [],
                [],
                bootstrap_entrypoint=False,
                gpu_device_ids=cfg.gpu_device_ids,
            )
            environment["FLOWMESH_FINISH_SENTINEL"] = plan.finish_sentinel.as_posix()
            bin_dir = _install_finish_helper(session_dir, plan.finish_sentinel)
            environment["PATH"] = f"{bin_dir.as_posix()}:{_DEFAULT_SESSION_PATH}"
            authorized_keys = session_dir / "authorized_keys"
            rendered, exported = _render_authorized_keys(
                cfg.authorized_keys, environment
            )
            authorized_keys.write_text(rendered, encoding="utf-8")
            # sshd opens this file as the session user, not as itself, so a
            # root-owned 0600 would deny every login.
            authorized_keys.chmod(0o644)
            process, port, log_path = _start_sshd(
                sshd_path, session_dir, host_key, authorized_keys, identity, exported
            )
        except Exception:
            if plan is not None:
                _discard_paths(plan)
            if identity is not None:
                identity.release()
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        return ProcessSession(
            backend=self,
            process=process,
            port=port,
            session_dir=session_dir,
            log_path=log_path,
            plan=plan,
            cfg=cfg,
            identity=identity,
        )

    def _build_paths(
        self, request: SessionRequest, session_dir: Path, identity: SessionIdentity
    ) -> "ProcessSessionPaths":
        """Place inputs and the output directory at their requested paths.

        With no mount namespace the requested absolute paths have to be created
        for real, so a worker that cannot write them fails loudly here rather
        than handing the user a session with silently missing data.
        """
        cfg = request.cfg
        used_mount_paths: set[str] = set()
        input_links: list[Path] = []
        staged_inputs_dir: Path | None = None
        output_created = False

        if request.resolved_inputs:
            staged_inputs_dir = stage_inputs_locally(
                request.resolved_inputs, request.session_id
            )
            identity.own(staged_inputs_dir, recursive=True)
            for resolved in request.resolved_inputs:
                reserve_mount_path(used_mount_paths, resolved.mount_path)
                target = Path(resolved.mount_path)
                _ensure_parent_dir(target, "inputs[].mountPath")
                if target.exists() or target.is_symlink():
                    raise ExecutionError(
                        f"SSH input mountPath {resolved.mount_path} already exists "
                        "on this worker"
                    )
                target.symlink_to(staged_inputs_dir / resolved.task_id)
                input_links.append(target)

        output_path: Path | None = None
        if cfg.output is not None:
            mount_path = normalize_mount_path(
                cfg.output.mount_path, field_name="sshOutput.mountPath"
            )
            reserve_mount_path(used_mount_paths, mount_path)
            output_path = Path(mount_path)
            _ensure_parent_dir(output_path, "sshOutput.mountPath")
            output_created = not output_path.exists()
            output_path.mkdir(parents=True, exist_ok=True)
            identity.own(output_path)

        # The sentinel lives under session_dir either way, so it cannot survive
        # into the next session and end it the moment it starts.
        sentinel_dir = identity.home if identity.isolates_from_worker else session_dir
        return ProcessSessionPaths(
            finish_sentinel=sentinel_dir / ".flowmesh_finish",
            staged_inputs_dir=staged_inputs_dir,
            input_links=input_links,
            output_path=output_path,
            output_created=output_created,
        )


@dataclass(slots=True)
class ProcessSessionPaths:
    """Absolute paths this session materialized outside its own directory."""

    finish_sentinel: Path
    staged_inputs_dir: Path | None
    input_links: list[Path]
    output_path: Path | None
    output_created: bool


def _discard_paths(plan: ProcessSessionPaths) -> None:
    for link in plan.input_links:
        try:
            link.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove SSH input link %s", link, exc_info=True)
    if plan.staged_inputs_dir is not None:
        shutil.rmtree(plan.staged_inputs_dir, ignore_errors=True)
    if plan.output_path is not None and plan.output_created:
        shutil.rmtree(plan.output_path, ignore_errors=True)


class ProcessSession(SSHSession):
    """An SSH session running as an sshd process beside the worker."""

    def __init__(
        self,
        backend: ProcessSessionBackend,
        process: subprocess.Popen[bytes],
        port: int,
        session_dir: Path,
        log_path: Path,
        plan: ProcessSessionPaths,
        cfg: SSHConfig,
        identity: SessionIdentity,
    ) -> None:
        self._backend = backend
        self._process = process
        self._port = port
        self._session_dir = session_dir
        self._log_path = log_path
        self._plan = plan
        self._cfg = cfg
        self.identity = identity

    def login_user(self) -> str:
        return self.identity.name

    def wait_ready(self, timeout_sec: float = 30.0) -> int:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                raise ExecutionError(
                    f"sshd exited (code {exit_code}) before the SSH session became "
                    f"ready.{self._log_tail()}"
                )
            if is_ssh_ready("127.0.0.1", self._port):
                return self._port
            time.sleep(0.5)
        raise ExecutionError(
            f"Timed out waiting for SSH readiness on port {self._port}."
            f"{self._log_tail()}"
        )

    def poll(self) -> int | None:
        return self._process.poll()

    def finish_requested(self) -> bool:
        return self._plan.finish_sentinel.exists()

    def established_connections(self) -> int | None:
        proc_net_tcp = read_local_proc_net_tcp()
        if proc_net_tcp is None:
            return None
        return count_established_connections(proc_net_tcp, self._port)

    def output_size_bytes(self) -> int | None:
        if (output_path := self._plan.output_path) is None:
            return None
        return path_size_bytes(output_path)

    def collect_output(self, destination: Path) -> None:
        if (output_path := self._plan.output_path) is None:
            return
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(output_path, destination, dirs_exist_ok=True)

    def stop(self, timeout_sec: float) -> None:
        process = self._process
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                logger.warning("sshd did not exit after SIGKILL")

    def cleanup(self) -> None:
        try:
            self.stop(_TERMINATE_GRACE_SEC)
            _discard_paths(self._plan)
            self.identity.release()
            shutil.rmtree(self._session_dir, ignore_errors=True)
        finally:
            # A failure above must not strand the worker refusing every later
            # session; the stale-account sweep in prepare() is the backstop.
            self._backend._release(self)

    def save_logs(self, out_dir: Path) -> None:
        try:
            if not self._log_path.exists():
                return
            logs_dir = out_dir / ARTIFACTS_DIR / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self._log_path, logs_dir / "sshd.log")
        except OSError:
            logger.debug("Failed to capture sshd logs", exc_info=True)

    def _log_tail(self, max_chars: int = 2000) -> str:
        text = _read_log(self._log_path)
        return f"\nsshd output:\n{text[-max_chars:]}" if text else ""


def _start_sshd(
    sshd_path: str,
    session_dir: Path,
    host_key: Path,
    authorized_keys: Path,
    identity: SessionIdentity,
    exported_env: list[str],
) -> tuple[subprocess.Popen[bytes], int, Path]:
    """Start sshd, retrying on another port when it loses the race to bind.

    ``_pick_free_port`` has to release the port before sshd claims it, so the
    kernel can hand it to something else in between.
    """
    log_path = session_dir / "sshd.log"
    config_path = session_dir / "sshd_config"
    detail = ""
    for _ in range(_PORT_ATTEMPTS):
        port = _pick_free_port()
        config_path.write_text(
            _render_sshd_config(
                port=port,
                session_dir=session_dir,
                host_key=host_key,
                authorized_keys=authorized_keys,
                login_user=identity.name,
                exported_env=exported_env,
            ),
            encoding="utf-8",
        )
        config_path.chmod(0o600)
        process = _spawn_sshd(sshd_path, config_path, log_path)
        deadline = time.time() + _READY_PROBE_SEC
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if is_ssh_ready("127.0.0.1", port):
                return process, port, log_path
            time.sleep(0.05)
        if process.poll() is None:
            return process, port, log_path
        detail = _read_log(log_path)
        if not any(marker in detail.lower() for marker in _BIND_FAILURE_MARKERS):
            raise ExecutionError(f"sshd exited immediately.\nsshd output:\n{detail}")
        logger.info("sshd could not bind port %d; retrying on another port", port)
    raise ExecutionError(
        f"sshd could not bind a free port after {_PORT_ATTEMPTS} attempts."
        f"\nsshd output:\n{detail}"
    )


def _spawn_sshd(
    sshd_path: str, config_path: Path, log_path: Path
) -> subprocess.Popen[bytes]:
    log_handle = log_path.open("wb")
    try:
        return subprocess.Popen(  # nosec B603 - argv list, no shell=True, absolute path via find_sshd()
            [sshd_path, "-D", "-e", "-f", config_path.as_posix()],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_sanitized_spawn_env(),
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        raise ExecutionError(f"Failed to start sshd: {exc}") from exc


def _generate_host_key(keygen_path: str, host_key: Path) -> None:
    result = subprocess.run(  # nosec B603 - argv list, no shell=True, absolute path via shutil.which()
        [keygen_path, "-q", "-t", "ed25519", "-N", "", "-f", host_key.as_posix()],
        capture_output=True,
        timeout=_KEYGEN_TIMEOUT_SEC,
        env=_sanitized_spawn_env(),
        check=False,
    )
    if result.returncode != 0 or not host_key.exists():
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutionError(f"Failed to generate SSH host key: {detail}")


def _install_finish_helper(session_dir: Path, sentinel: Path) -> Path:
    """Give the session the ``flowmesh-finish`` command the Docker image ships.

    It lives in a per-session directory rather than ``/usr/local/bin`` so that a
    non-root worker can install it too, and so it leaves with the session.
    ``0711`` is enough for a PATH lookup, which stats candidates rather than
    listing the directory.
    """
    bin_dir = session_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    helper = bin_dir / "flowmesh-finish"
    helper.write_text(
        "#!/bin/sh\n"
        "set -e\n"
        f'touch "{sentinel.as_posix()}"\n'
        'echo "FlowMesh finish requested; the SSH session will close shortly."\n',
        encoding="utf-8",
    )
    helper.chmod(0o755)
    bin_dir.chmod(0o711)
    return bin_dir


def _sanitized_spawn_env() -> dict[str, str]:
    """Environment for helper processes, carrying none of the worker's secrets.

    sshd would otherwise inherit the worker's environment, and ``execve`` resets
    the dumpable flag, so the session could read the task token and every API
    key straight out of ``/proc/<sshd>/environ``.
    """
    env = {key: value for key in _SPAWN_ENV_KEYS if (value := os.environ.get(key))}
    env.setdefault("PATH", os.defpath)
    return env


def _hide_worker_environ() -> bool:
    """Make this process's ``/proc`` entry unreadable to its own uid.

    Clearing the dumpable flag reassigns ``/proc/<pid>`` to root, which is what
    stops a same-uid session from reading the worker's environment or attaching
    to it.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return bool(libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0) == 0)
    except (OSError, AttributeError):
        return False


def _render_sshd_config(
    port: int,
    session_dir: Path,
    host_key: Path,
    authorized_keys: Path,
    login_user: str,
    exported_env: list[str] | None = None,
) -> str:
    permit_env = ",".join(exported_env) if exported_env else "no"
    return "\n".join(
        (
            f"Port {port}",
            "ListenAddress 0.0.0.0",
            f"HostKey {host_key.as_posix()}",
            f"PidFile {(session_dir / 'sshd.pid').as_posix()}",
            f"AuthorizedKeysFile {authorized_keys.as_posix()}",
            f"AllowUsers {login_user}",
            "PasswordAuthentication no",
            "KbdInteractiveAuthentication no",
            "PubkeyAuthentication yes",
            "PermitRootLogin prohibit-password",
            f"PermitUserEnvironment {permit_env}",
            "StrictModes no",
            "UsePAM no",
            "PrintMotd no",
            "AllowTcpForwarding no",
            "X11Forwarding no",
            "AllowAgentForwarding no",
            "GatewayPorts no",
            "Subsystem sftp internal-sftp",
            "",
        )
    )


def _render_authorized_keys(
    authorized_keys: list[str], environment: dict[str, str]
) -> tuple[str, list[str]]:
    """Render authorized_keys, carrying session env as per-key options.

    sshd does not inherit the worker's environment into a login shell, so the
    values the session is supposed to see (``CUDA_VISIBLE_DEVICES`` above all)
    travel as ``environment=`` options on each key. Returns the rendered file
    and the names actually exported, which ``PermitUserEnvironment`` must list.
    """
    exported = [
        name
        for name, value in sorted(environment.items())
        if _is_safe_env_entry(name, value)
    ]
    options = ",".join(f'environment="{name}={environment[name]}"' for name in exported)
    lines = [
        f"{options} {key}" if options else key
        for raw_key in authorized_keys
        if (key := raw_key.strip())
    ]
    return ("\n".join(lines) + "\n" if lines else "", exported if lines else [])


def _is_safe_env_entry(name: str, value: str) -> bool:
    if not _ENV_NAME_RE.match(name):
        logger.warning("Dropping SSH session env var with unsupported name %r", name)
        return False
    if any(ch in value for ch in _ENV_VALUE_FORBIDDEN):
        logger.warning("Dropping SSH session env var %s: unsupported value", name)
        return False
    return True


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _read_log(log_path: Path, max_chars: int = 4000) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace").strip()[
            -max_chars:
        ]
    except OSError:
        return ""


def _ensure_parent_dir(path: Path, field_name: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExecutionError(
            f"Cannot create {field_name} {path.as_posix()} on this worker: {exc}. "
            "Process-mode SSH sessions have no mount namespace, so the requested "
            "path has to be writable by the worker itself."
        ) from exc
