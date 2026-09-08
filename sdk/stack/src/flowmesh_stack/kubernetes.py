"""kubectl helpers for the FlowMesh Kubernetes stack backend."""

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .env import parse_env_file
from .manifests import render_manifests

MANAGED_LABEL = "flowmesh.io/managed"
"""Label carried by worker pods the supervisor created."""

STACK_LABEL = "app.kubernetes.io/part-of"
"""Label carried by every stack resource."""

STACK_LABEL_VALUE = "flowmesh"


class KubectlError(RuntimeError):
    """Raised when a kubectl command fails."""


def ensure_kubectl_available() -> str:
    """Return the absolute kubectl path, raising when it is not installed."""
    path = shutil.which("kubectl")
    if path is None:
        raise KubectlError("kubectl is required but was not found in PATH")
    return path


def kubectl(
    args: list[str],
    namespace: str | None = None,
    context: str | None = None,
    kubeconfig: str | None = None,
    stdin: str | None = None,
    env: Mapping[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run kubectl with the provided arguments."""
    cmd = [ensure_kubectl_available()]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    if context:
        cmd += ["--context", context]
    if namespace:
        cmd += ["--namespace", namespace]
    cmd += args

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)

    return (
        subprocess.run(  # nosec B603: argv list, no shell, absolute path via which().
            cmd,
            check=False,
            input=stdin,
            capture_output=capture_output,
            text=True,
            env=merged_env,
        )
    )


@dataclass
class KubernetesStack:
    """Applies and inspects the FlowMesh stack in a Kubernetes namespace."""

    manifests: list[Path]
    """Manifest assets rendered for every apply."""
    namespace: str
    """Namespace the stack is deployed into."""
    load_env: Callable[[Path], None]
    """Callback that loads and resolves env-file values before operations run."""
    context: str | None = None
    """kubectl context to target."""
    kubeconfig: str | None = None
    """kubeconfig file to use instead of the default."""
    rollout_targets: list[str] = field(default_factory=list)
    """Workloads that ``up`` waits on before reporting success."""

    def render(self, env_file: Path) -> str:
        """Render the stack manifests using the resolved environment.

        Substitution reads the resolved process environment; the values handed
        to the cluster come from the environment file alone, so nothing else in
        the operator's shell is copied into the namespace.
        """
        self.load_env(env_file)
        return render_manifests(
            self.manifests, dict(os.environ), parse_env_file(env_file)
        )

    def _run(
        self, args: list[str], stdin: str | None = None, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return kubectl(
            args,
            namespace=self.namespace,
            context=self.context,
            kubeconfig=self.kubeconfig,
            stdin=stdin,
            capture_output=capture_output,
        )

    def apply(self, env_file: Path) -> subprocess.CompletedProcess[str]:
        """Apply the rendered stack manifests."""
        return self._run(["apply", "-f", "-"], stdin=self.render(env_file))

    def delete(
        self, env_file: Path, ignore_not_found: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Delete the resources described by the rendered stack manifests."""
        args = ["delete", "-f", "-"]
        if ignore_not_found:
            args.append("--ignore-not-found")
        return self._run(args, stdin=self.render(env_file))

    def rollout_status(self, target: str, timeout: str) -> subprocess.CompletedProcess:
        """Wait for a workload to finish rolling out."""
        return self._run(["rollout", "status", target, f"--timeout={timeout}"])

    def rollout_restart(self, target: str) -> subprocess.CompletedProcess[str]:
        """Restart a workload in place."""
        return self._run(["rollout", "restart", target])

    def logs(self, target: str, follow: bool = True) -> subprocess.CompletedProcess:
        """Stream logs for a workload or pod."""
        args = ["logs", target, "--all-containers"]
        if follow:
            args.append("-f")
        return self._run(args)

    def status(self) -> subprocess.CompletedProcess[str]:
        """Show stack pods."""
        return self._run(
            ["get", "pods", "-l", f"{STACK_LABEL}={STACK_LABEL_VALUE}", "-o", "wide"]
        )

    def worker_status(self) -> subprocess.CompletedProcess[str]:
        """Show pods the supervisor created for workers."""
        return self._run(["get", "pods", "-l", f"{MANAGED_LABEL}=true", "-o", "wide"])

    def delete_workers(self) -> subprocess.CompletedProcess[str]:
        """Delete every supervisor-managed worker pod in the namespace."""
        return self._run(["delete", "pods", "-l", f"{MANAGED_LABEL}=true"])

    def delete_volumes(self) -> subprocess.CompletedProcess[str]:
        """Delete the stack's persistent volume claims."""
        return self._run(["delete", "pvc", "-l", f"{STACK_LABEL}={STACK_LABEL_VALUE}"])
