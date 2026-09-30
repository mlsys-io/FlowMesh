"""Process session backend: sshd as a process beside the worker.

For workers with no Docker socket, which cannot create a sibling container to
put a session in.

Three consequences follow from having no container around the session and are
enforced here rather than assumed:

* **One session per worker.** Sessions sharing a worker would share its
  filesystem and process namespace, so a second concurrent session is refused.
* **No worker-side resource cap.** ``SSH_MAX_CPU`` / ``SSH_MAX_MEMORY`` /
  ``SSH_MAX_PIDS`` need cgroup control the worker does not have over itself;
  the size of the rented box is the cap.
* **Isolation depends on the worker's own privileges.** A root worker gives
  each session its own account, so the session cannot reach the worker's
  credentials, and denies that account the worker's state (results, caches,
  home) through ACLs. A non-root worker can only authenticate its own account,
  and the session inherits everything that account can read.

Root acts on paths the session can write, so it never follows a link in one.
The session's inputs and output live in its own root-owned directory, and the
requested mount paths are links to them, created component by component under a
mount root that is emptied before and after every session.
"""

import ctypes
import errno
import fcntl
import logging
import os
import re
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shared.schemas.worker import SSHBackendName
from shared.tasks.worker_message import WorkerHardware
from shared.utils import parse_float_env
from shared.utils.manifest import ARTIFACTS_DIR
from worker.config import WorkerConfig

from ...base_executor import ExecutionError
from .. import acl
from ..base import (
    SessionRequest,
    SSHSession,
    SSHSessionBackend,
    count_established_connections,
    is_ssh_ready,
    iter_tree,
    path_size_bytes,
    read_local_proc_net_tcp,
    resolve_tailnet_address,
)
from ..config import (
    SAFE_MOUNT_ROOT,
    STOP_TIMEOUT_SEC,
    SSHConfig,
    normalize_mount_path,
    reserve_mount_path,
)
from ..inputs import stage_inputs_locally
from ..session_identity import (
    PRIVSEP_DIR,
    SESSION_DIR_PREFIX,
    SessionIdentity,
    live_session_accounts,
    reap_stale_accounts,
    remove_tree,
    resolve_identity,
)

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
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC
_COPY_CHUNK = 1024 * 1024
_BACKEND_LOCK_NAME = "flowmesh-ssh-process.lock"
# Paths every session needs; a denied root covering one would break it.
_SESSION_REQUIRED_PATHS = (
    Path(SAFE_MOUNT_ROOT),
    PRIVSEP_DIR,
    Path("/usr"),
    Path("/bin"),
    Path("/etc"),
    Path("/lib"),
)
# Every path-typed ``WorkerConfig`` field belongs to exactly one of these.
DENIED_CONFIG_FIELDS = ("results_dir", "hb_file", "state_dirs")
ALLOWED_CONFIG_FIELDS: tuple[str, ...] = ()
_MAX_LINK_HOPS = 40
_MOUNTINFO = Path("/proc/self/mountinfo")
_OCTAL_ESCAPE_RE = re.compile(r"\\([0-7]{3})")
_backend_lock_fd: int | None = None
_backend_lock_mutex = threading.Lock()


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
    name = SSHBackendName.PROCESS
    supports_noninteractive = False

    def __init__(
        self, config: WorkerConfig, hardware: WorkerHardware | None = None
    ) -> None:
        super().__init__(config, hardware)
        self._lock = threading.Lock()
        self._active: ProcessSession | None = None

    @classmethod
    def is_available(cls, config: WorkerConfig) -> bool:
        if os.getuid() != 0 and not config.enable_unisolated_ssh_session:
            logger.info(
                "Process SSH backend unavailable: this worker is not root, so a "
                "session would run under its own account and could read its "
                "credentials. Set ENABLE_UNISOLATED_SSH_SESSION=true to accept "
                "that and serve sessions anyway."
            )
            return False
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
        return cls._isolation_ready(config)

    @classmethod
    def _isolation_ready(cls, config: WorkerConfig) -> bool:
        """Whether this worker can keep a session away from what it must not reach."""
        if os.getuid() == 0 and not _acl_ready(config):
            return False
        if not _acquire_backend_lock():
            logger.info(
                "Process SSH backend unavailable: another worker sharing this root "
                "filesystem already serves process-mode sessions, and they would "
                "share %s",
                SAFE_MOUNT_ROOT.as_posix(),
            )
            return False
        return True

    def prepare(self) -> None:
        active = self._active_account_name()
        reap_stale_accounts(keep=active)
        if active is None:
            try:
                _reset_mount_root(Path(SAFE_MOUNT_ROOT), create=False)
            except OSError:
                logger.warning(
                    "Could not clear %s", SAFE_MOUNT_ROOT.as_posix(), exc_info=True
                )
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

    def _default_session_host(self) -> str:
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
            if lingering := live_session_accounts():
                logger.warning(
                    "Refusing an SSH session: processes of earlier session accounts "
                    "%s are still running",
                    ", ".join(lingering),
                )
                raise ExecutionError(
                    "This worker still has processes of an earlier SSH session "
                    "running; refusing a new session until they are gone.",
                    retryable=True,
                )
            session = self._create_session(request)
            self._active = session
        return session

    def teardown(self, owner: str) -> None:
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
            tempfile.mkdtemp(prefix=f"{SESSION_DIR_PREFIX}{request.session_id[:8]}-")
        )
        session_dir.chmod(0o700)
        identity: SessionIdentity | None = None
        plan: ProcessSessionPaths | None = None
        try:
            denied_roots = self._claim_denied_roots(session_dir)
            identity = resolve_identity(request.session_id, session_dir, denied_roots)
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
                sshd_path,
                session_dir,
                host_key,
                authorized_keys,
                identity,
                exported,
                self.session_bind_host(cfg.access_mode),
            )
        except Exception:
            if plan is not None:
                _discard_paths(plan)
            if identity is not None:
                identity.release()
            remove_tree(session_dir)
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

    def _claim_denied_roots(self, session_dir: Path) -> list[Path]:
        """The worker-state roots to deny this session, each ready for an ACL.

        A missing root is created, so that one appearing mid-session is covered
        too.
        """
        if os.getuid() != 0:
            return []
        if problem := _state_problem(self._config):
            raise ExecutionError(f"Refusing the SSH session: {problem}", retryable=True)
        roots = denied_roots(self._config)
        for root in roots:
            if session_dir.is_relative_to(root):
                raise ExecutionError(
                    f"Refusing the SSH session: its directory {session_dir} is inside "
                    f"the worker state {root} it must be denied",
                    retryable=True,
                )
            if not os.path.lexists(root):
                try:
                    _create_state_root(root)
                except OSError as exc:
                    raise ExecutionError(
                        f"Cannot create worker state {root}: {exc}", retryable=True
                    ) from exc
        if problem := _state_problem(self._config):
            raise ExecutionError(f"Refusing the SSH session: {problem}", retryable=True)
        return denied_roots(self._config)

    def _build_paths(
        self, request: SessionRequest, session_dir: Path, identity: SessionIdentity
    ) -> "ProcessSessionPaths":
        """Stage inputs and the output directory, and link their mount paths.

        Both live under ``session_dir``, which only root can change, so the
        session can never swap them for something root would then follow. The
        mount paths are links to them in a mount root emptied first, so nothing
        an earlier session left there is ever walked through.
        """
        cfg = request.cfg
        used_mount_paths: set[str] = set()
        links: list[tuple[str, Path]] = []

        if request.resolved_inputs:
            staged_inputs_dir = session_dir / "inputs"
            identity.make_dir(staged_inputs_dir, 0o700)
            stage_inputs_locally(
                request.resolved_inputs,
                request.session_id,
                staging_dir=staged_inputs_dir,
            )
            identity.own(staged_inputs_dir, recursive=True)
            for resolved in request.resolved_inputs:
                reserve_mount_path(used_mount_paths, resolved.mount_path)
                links.append(
                    (resolved.mount_path, staged_inputs_dir / resolved.task_id)
                )

        output_path: Path | None = None
        if cfg.output is not None:
            mount_path = normalize_mount_path(
                cfg.output.mount_path, field_name="sshOutput.mountPath"
            )
            reserve_mount_path(used_mount_paths, mount_path)
            output_path = session_dir / "output"
            identity.make_dir(output_path, 0o700)
            identity.own(output_path)
            links.append((mount_path, output_path))

        mount_root = Path(SAFE_MOUNT_ROOT)
        if links:
            try:
                _reset_mount_root(mount_root, create=True)
            except OSError as exc:
                raise ExecutionError(
                    f"Cannot prepare {mount_root.as_posix()} on this worker: {exc}. "
                    "Process-mode SSH sessions have no mount namespace, so the "
                    "mount root has to be writable by the worker itself.",
                    retryable=True,
                ) from exc
            for mount_path, target in links:
                _link_mount_path(mount_root, mount_path, target)

        # The sentinel lives under session_dir either way, so it cannot survive
        # into the next session and end it the moment it starts.
        sentinel_dir = identity.home if identity.isolates_from_worker else session_dir
        return ProcessSessionPaths(
            finish_sentinel=sentinel_dir / ".flowmesh_finish",
            output_path=output_path,
            mount_root=mount_root if links else None,
        )


@dataclass(slots=True)
class ProcessSessionPaths:
    """Where this session's data lives, and whether it linked any mount paths."""

    finish_sentinel: Path
    output_path: Path | None
    mount_root: Path | None


def _discard_paths(plan: ProcessSessionPaths) -> None:
    if plan.mount_root is None:
        return
    try:
        _reset_mount_root(plan.mount_root, create=False)
    except OSError:
        logger.debug("Failed to clear %s", plan.mount_root, exc_info=True)


def _reset_mount_root(root: Path, create: bool) -> None:
    """Empty ``root`` without following a link, leaving it root-owned ``0755``.

    ``root`` itself is kept rather than replaced when it is a directory, since
    it may be a mount point.
    """
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        if not create:
            return
        root.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(root, 0o755)
        info = os.lstat(root)
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(root)
        if not create:
            return
        os.mkdir(root, 0o755)
    _refuse_nested_mounts(root)
    fd = os.open(root, _DIR_FLAGS)
    try:
        with os.scandir(fd) as scanner:
            entries = list(scanner)
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.name, dir_fd=fd)
            else:
                os.unlink(entry.name, dir_fd=fd)
        if os.geteuid() == 0:
            os.fchown(fd, 0, 0)
        os.fchmod(fd, 0o755)  # nosec B103 - the session must traverse it
    finally:
        os.close(fd)


def _refuse_nested_mounts(root: Path) -> None:
    """Refuse to empty ``root`` when a filesystem is mounted anywhere below it.

    Everything under the mount root is the backend's own, so a mount there is
    an operator's, and emptying the root would delete what it holds. A bind
    mount from the same filesystem keeps the device number, so the kernel's
    mount table is what finds it; the device check covers a worker that cannot
    read that table.
    """
    for mount_point in _mount_points():
        if mount_point != root and mount_point.is_relative_to(root):
            raise OSError(
                errno.EBUSY, "a filesystem is mounted below the mount root", mount_point
            )
    device = os.lstat(root).st_dev
    for parent, dirs, _ in os.walk(root, followlinks=False):
        for name in dirs:
            path = os.path.join(parent, name)
            if os.lstat(path).st_dev != device:
                raise OSError(
                    errno.EBUSY, "a filesystem is mounted below the mount root", path
                )


def _mount_points() -> list[Path]:
    """Mount points in this process's mount namespace, or none when unreadable."""
    try:
        text = _MOUNTINFO.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    points: list[Path] = []
    for line in text.splitlines():
        fields = line.split(" ")
        if len(fields) > 4:
            points.append(Path(_OCTAL_ESCAPE_RE.sub(_unescape_octal, fields[4])))
    return points


def _unescape_octal(match: re.Match[str]) -> str:
    return chr(int(match.group(1), 8))


def _link_mount_path(root: Path, mount_path: str, target: Path) -> None:
    """Create ``mount_path`` as a link to ``target``, never following a link.

    Each missing component below ``root`` is created relative to its parent's
    descriptor, and an existing one is only entered if it is a real directory.
    """
    parts = PurePosixPath(mount_path).relative_to(PurePosixPath(root)).parts
    if not parts:
        raise ExecutionError(
            f"mountPath {mount_path} must name a path below {root.as_posix()} on "
            "this worker"
        )
    fd = os.open(root, _DIR_FLAGS)
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                raise ExecutionError(
                    f"mountPath {mount_path} conflicts with another mountPath"
                ) from exc
            os.close(fd)
            fd = child
            os.fchmod(fd, 0o755)  # nosec B103 - the session must traverse it
        try:
            os.symlink(target, parts[-1], dir_fd=fd)
        except FileExistsError as exc:
            raise ExecutionError(
                f"mountPath {mount_path} conflicts with another mountPath"
            ) from exc
    finally:
        os.close(fd)


def denied_roots(config: WorkerConfig) -> list[Path]:
    """Return the worker state a session is denied, resolved to the paths
    ``setfacl`` acts on."""
    return list(
        dict.fromkeys(Path(os.path.realpath(path)) for path in _state_paths(config))
    )


def _state_paths(config: WorkerConfig) -> list[Path]:
    """Return each path in ``DENIED_CONFIG_FIELDS`` as the worker uses it, made
    absolute, with the heartbeat file replaced by its directory."""
    paths: list[Path] = []
    for field_name in DENIED_CONFIG_FIELDS:
        value = getattr(config, field_name)
        for configured in value if isinstance(value, tuple) else (value,):
            path = Path(configured).absolute()
            # Denying only the file would still let a session list its name,
            # which contains the worker token.
            paths.append(path.parent if field_name == "hb_file" else path)
    return paths


def _create_state_root(root: Path) -> None:
    """Create ``root`` as ``0700`` and each missing parent as ``0755``, never
    following a link."""
    parts = root.parts[1:]
    fd = os.open("/", _DIR_FLAGS)
    try:
        for index, part in enumerate(parts):
            try:
                os.mkdir(part, 0o700 if index == len(parts) - 1 else 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
    finally:
        os.close(fd)


def _state_problem(config: WorkerConfig) -> str | None:
    """Why this worker's state cannot be denied to a session safely, if it cannot."""
    roots = denied_roots(config)
    for path in _state_paths(config):
        if problem := _path_problem(path, roots):
            return problem
    for root in roots:
        if blocked := _required_path_under(root):
            return f"denying {root} would also deny {blocked}"
    return None


def _path_problem(path: Path, denied: Sequence[Path] = ()) -> str | None:
    """Why a session could replace ``path`` or a directory on the way to it, if
    it could.

    Every directory that resolving ``path`` looks a name up in is checked,
    those a link leads through included, except one at or below a root in
    ``denied``, which the session cannot search. A session can rename any entry
    of a directory it can write; in a sticky one only its own, so there the
    entry must be a directory the worker owns, not a link.
    """
    directory = Path("/")
    pending = list(path.parts[1:])
    hops = 0
    while pending:
        name = pending.pop(0)
        if name == "..":
            directory = directory.parent
            continue
        entry = directory / name
        try:
            dir_mode = os.stat(directory).st_mode
            info: os.stat_result | None = os.lstat(entry)
        except FileNotFoundError:
            info = None
        except OSError as exc:
            return f"cannot inspect {entry} on the way to worker state {path}: {exc}"
        if dir_mode & stat.S_IWOTH and not any(
            directory.is_relative_to(root) for root in denied
        ):
            if not dir_mode & stat.S_ISVTX:
                return (
                    f"{directory}, on the way to worker state {path}, is "
                    "world-writable, so a session could replace what it holds"
                )
            if info is not None and (
                info.st_uid != os.geteuid() or stat.S_ISLNK(info.st_mode)
            ):
                return (
                    f"{entry}, on the way to worker state {path}, sits in a shared "
                    "directory but is not a directory this worker owns"
                )
        if info is None:
            return None
        if stat.S_ISLNK(info.st_mode):
            hops += 1
            if hops > _MAX_LINK_HOPS:
                return f"too many links on the way to worker state {path}"
            target = PurePosixPath(os.readlink(entry))
            if target.is_absolute():
                directory = Path("/")
                pending[:0] = target.parts[1:]
            else:
                pending[:0] = target.parts
            continue
        directory = entry
    return None


def _acl_ready(config: WorkerConfig) -> bool:
    if not acl.tools_available():
        logger.info(
            "Process SSH backend unavailable: setfacl/getfacl are missing, so a "
            "session could not be denied this worker's state (install the acl "
            "package in the worker image)"
        )
        return False
    if problem := _state_problem(config):
        logger.info("Process SSH backend unavailable: %s", problem)
        return False
    for root in denied_roots(config):
        try:
            acl.probe(_probe_dir(root))
        except (OSError, ExecutionError) as exc:
            logger.info(
                "Process SSH backend unavailable: cannot deny sessions %s with an "
                "ACL: %s",
                root,
                exc,
            )
            return False
    return True


def _required_path_under(root: Path) -> Path | None:
    """A path every session needs that denying ``root`` would also deny."""
    for required in (Path(tempfile.gettempdir()), *_SESSION_REQUIRED_PATHS):
        if required == root or required.is_relative_to(root):
            return required
    return None


def _probe_dir(root: Path) -> Path:
    """The existing directory whose filesystem will hold ``root``."""
    candidate = root
    while not candidate.is_dir():
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return candidate


def _acquire_backend_lock() -> bool:
    """Take the process-backend lock file, in ``/run`` for root or the temp dir
    otherwise, for the life of the worker; return whether this worker holds it.

    Workers that see the same lock file share the mount root and session
    accounts, so only one of them serves sessions.
    """
    global _backend_lock_fd
    with _backend_lock_mutex:
        if _backend_lock_fd is not None:
            return True
        base = Path("/run") if os.geteuid() == 0 else Path(tempfile.gettempdir())
        try:
            fd = os.open(
                base / _BACKEND_LOCK_NAME,
                os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
        except OSError:
            logger.debug("Cannot open the process-backend lock", exc_info=True)
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                logger.debug("Cannot take the process-backend lock", exc_info=True)
            return False
        _backend_lock_fd = fd
        return True


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
        # Root must not follow a link the session placed here.
        return os.path.lexists(self._plan.finish_sentinel)

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
        """Copy the session's regular files into ``destination``.

        The session is ended first, so the tree holds still while root reads
        it. Links and special files are dropped, and a file the session does
        not own is too: a hard link to a file it cannot read would otherwise
        let root copy that file out for it.
        """
        if (output_path := self._plan.output_path) is None:
            return
        self.stop(_TERMINATE_GRACE_SEC)
        if not self.identity.terminate_processes():
            raise ExecutionError(
                "Could not stop the SSH session's processes to collect its output"
            )
        max_bytes = self._cfg.output.max_bytes if self._cfg.output else None
        destination.mkdir(parents=True, exist_ok=True)
        collected = 0
        try:
            for relative, entry, parent_fd in iter_tree(output_path):
                target = destination / relative
                if entry.is_dir(follow_symlinks=False):
                    target.mkdir(exist_ok=True)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    logger.debug("Not collecting %s: not a regular file", relative)
                    continue
                budget = None
                if max_bytes is not None and max_bytes >= 0:
                    budget = max_bytes - collected
                collected += _copy_owned_file(
                    parent_fd, entry.name, target, self.identity.uid, budget
                )
        except OSError as exc:
            raise ExecutionError(f"Failed to collect SSH output: {exc}") from exc

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
            remove_tree(self._session_dir)
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
            dir_fd = os.open(logs_dir, _DIR_FLAGS)
            try:
                out_fd = os.open("sshd.log", _WRITE_FLAGS, 0o644, dir_fd=dir_fd)
            finally:
                os.close(dir_fd)
            with os.fdopen(out_fd, "wb") as sink, self._log_path.open("rb") as source:
                shutil.copyfileobj(source, sink)
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
    bind_host: str,
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
                bind_host=bind_host,
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
    bind_host: str,
    exported_env: list[str] | None = None,
) -> str:
    permit_env = ",".join(exported_env) if exported_env else "no"
    return "\n".join(
        (
            f"Port {port}",
            f"ListenAddress {bind_host}",
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


def _copy_owned_file(
    parent_fd: int, name: str, target: Path, owner: int, budget: int | None
) -> int:
    """Copy ``name`` to ``target`` if it is a regular file ``owner`` owns.

    Refuses a file larger than ``budget`` bytes before writing any of it.
    Returns the bytes copied.
    """
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
    except OSError:
        logger.debug("Not collecting %s: cannot open it as a file", name)
        return 0
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != owner:
            logger.debug("Not collecting %s: not a file the session owns", name)
            return 0
        if budget is not None and info.st_size > budget:
            raise ExecutionError(
                f"Session output exceeded maxBytes by {info.st_size - budget} bytes"
            )
        with target.open("wb") as sink:
            shutil.copyfileobj(source, sink, _COPY_CHUNK)
        return info.st_size
