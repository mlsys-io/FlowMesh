"""HTTP translation for supervisor command responses."""

from fastapi import HTTPException, status

from shared.schemas.command import CommandErrorCode, CommandResponse

_STATUS_BY_ERROR_CODE: dict[CommandErrorCode, int] = {
    CommandErrorCode.INVALID_PAYLOAD: status.HTTP_400_BAD_REQUEST,
    CommandErrorCode.PROVIDER_UNAVAILABLE: status.HTTP_409_CONFLICT,
    CommandErrorCode.NOT_READY: status.HTTP_503_SERVICE_UNAVAILABLE,
    CommandErrorCode.CANCELLED: status.HTTP_503_SERVICE_UNAVAILABLE,
    CommandErrorCode.UNKNOWN_COMMAND: status.HTTP_501_NOT_IMPLEMENTED,
    CommandErrorCode.INTERNAL: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def command_error(
    resp: CommandResponse, default_detail: str = "Command failed"
) -> HTTPException:
    """Build the HTTP error matching a failed command response's error code.

    An unset or unrecognized code falls back to `500 Internal Server Error`.
    """
    error_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    if resp.error_code is not None:
        error_status = _STATUS_BY_ERROR_CODE.get(resp.error_code, error_status)
    return HTTPException(
        status_code=error_status, detail=resp.message or default_detail
    )


__all__ = ["command_error"]
