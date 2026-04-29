"""Stringification helpers for `ProfileSummary` in non-interactive contexts.

The SDK leaves rendering to the caller, but exposes a few convenience
adapters so downstream Python tools (notebooks, lumilake, ad-hoc scripts)
don't have to reimplement them.

Dataframe helpers require pandas, which is not in the default
`flowmesh-sdk` install. Install with the `dataframes` extra:
``pip install "flowmesh-sdk[dataframes]"``.
"""

from typing import Any

from shared.profile import ProfileSummary
from shared.profile import to_mermaid as _to_mermaid

try:
    import pandas as pd

    _PANDAS_OK = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False


_PANDAS_HINT = (
    "pandas is required for dataframe helpers; "
    'install with `pip install "flowmesh-sdk[dataframes]"`'
)


def hardware_dataframe(
    summary: ProfileSummary, *, on_critical_path: bool = False
) -> "pd.DataFrame":
    """Hardware-time table as a pandas DataFrame.

    Set `on_critical_path=True` to restrict to the critical path's
    hardware breakdown. Falls back to the e2e breakdown otherwise.
    """
    if not _PANDAS_OK:
        raise ImportError(_PANDAS_HINT)
    hw = (
        summary.critical_path.hardware_summary
        if on_critical_path and summary.critical_path is not None
        else summary.e2e_breakdown.hardware_summary
    )
    df = pd.DataFrame(
        {
            "event_type": hw.event_type,
            "count": hw.count,
            "total_hardware_time_seconds": hw.total_hardware_time_seconds,
            "avg_time_seconds": hw.avg_time_seconds,
            "min_time_seconds": hw.min_time_seconds,
            "max_time_seconds": hw.max_time_seconds,
        }
    )
    return df.sort_values(
        "total_hardware_time_seconds", ascending=False, na_position="last"
    ).reset_index(drop=True)


def network_dataframe(
    summary: ProfileSummary, *, on_critical_path: bool = False
) -> "pd.DataFrame":
    """Network-active-time table as a pandas DataFrame."""
    if not _PANDAS_OK:
        raise ImportError(_PANDAS_HINT)
    net = (
        summary.critical_path.network_summary
        if on_critical_path and summary.critical_path is not None
        else summary.e2e_breakdown.network_summary
    )
    df = pd.DataFrame(
        {
            "event_type": net.event_type,
            "count": net.count,
            "total_active_seconds": net.total_active_seconds,
            "avg_time_seconds": net.avg_time_seconds,
            "min_time_seconds": net.min_time_seconds,
            "max_time_seconds": net.max_time_seconds,
        }
    )
    return df.sort_values("total_active_seconds", ascending=False).reset_index(
        drop=True
    )


def critical_path_dataframe(summary: ProfileSummary) -> "pd.DataFrame":
    """Per-node active vs wait on the critical path."""
    if not _PANDAS_OK:
        raise ImportError(_PANDAS_HINT)
    if summary.critical_path is None:
        return pd.DataFrame(columns=["data_id", "active_seconds", "wait_seconds"])
    awb = summary.critical_path.active_wait_breakdown
    return pd.DataFrame(
        {
            "data_id": awb.data_id,
            "active_seconds": awb.active_seconds,
            "wait_seconds": awb.wait_seconds,
        }
    )


def to_mermaid(summary: ProfileSummary | dict[str, Any]) -> str:
    """Lineage DAG as Mermaid `graph TD` source."""
    if isinstance(summary, dict):
        summary = ProfileSummary.model_validate(summary)
    return _to_mermaid(summary)
