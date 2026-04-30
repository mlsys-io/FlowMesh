"""Client and server classes corresponding to protobuf-defined services."""

import warnings

import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2

from ...supervisor.v1 import supervisor_pb2 as supervisor_dot_v1_dot_supervisor__pb2

GRPC_GENERATED_VERSION = "1.71.2"
GRPC_VERSION = grpc.__version__
_version_not_supported = False
try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(
        GRPC_VERSION, GRPC_GENERATED_VERSION
    )
except ImportError:
    _version_not_supported = True
if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + f" but the generated code in supervisor/v1/supervisor_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class SupervisorStub(object):
    """Supervisor streams tasks to workers and accepts events from them."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterWorker = channel.unary_unary(
            "/supervisor.v1.Supervisor/RegisterWorker",
            request_serializer=supervisor_dot_v1_dot_supervisor__pb2.RegisterRequest.SerializeToString,
            response_deserializer=supervisor_dot_v1_dot_supervisor__pb2.RegisterResponse.FromString,
            _registered_method=True,
        )
        self.StreamTasks = channel.unary_stream(
            "/supervisor.v1.Supervisor/StreamTasks",
            request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            response_deserializer=supervisor_dot_v1_dot_supervisor__pb2.DispatchMessage.FromString,
            _registered_method=True,
        )
        self.PushEvents = channel.stream_unary(
            "/supervisor.v1.Supervisor/PushEvents",
            request_serializer=supervisor_dot_v1_dot_supervisor__pb2.EventMessage.SerializeToString,
            response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            _registered_method=True,
        )
        self.PushLogs = channel.stream_unary(
            "/supervisor.v1.Supervisor/PushLogs",
            request_serializer=supervisor_dot_v1_dot_supervisor__pb2.LogMessage.SerializeToString,
            response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            _registered_method=True,
        )
        self.FetchData = channel.unary_unary(
            "/supervisor.v1.Supervisor/FetchData",
            request_serializer=supervisor_dot_v1_dot_supervisor__pb2.FetchDataRequest.SerializeToString,
            response_deserializer=supervisor_dot_v1_dot_supervisor__pb2.FetchDataResponse.FromString,
            _registered_method=True,
        )
        self.PublishData = channel.unary_unary(
            "/supervisor.v1.Supervisor/PublishData",
            request_serializer=supervisor_dot_v1_dot_supervisor__pb2.PublishDataRequest.SerializeToString,
            response_deserializer=supervisor_dot_v1_dot_supervisor__pb2.PublishDataResponse.FromString,
            _registered_method=True,
        )


class SupervisorServicer(object):
    """Supervisor streams tasks to workers and accepts events from them."""

    def RegisterWorker(self, request, context):
        """Register a worker and return its worker id + auth token."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def StreamTasks(self, request, context):
        """Stream tasks assigned to a worker token."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PushEvents(self, request_iterator, context):
        """Accept a stream of worker/task events for relay to the supervisor."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PushLogs(self, request_iterator, context):
        """Accept a stream of task-scoped logs for relay to the supervisor."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def FetchData(self, request, context):
        """Read a cross-task data payload from the node-local cache."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def PublishData(self, request, context):
        """Publish a cross-task data payload to the node-local cache."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_SupervisorServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "RegisterWorker": grpc.unary_unary_rpc_method_handler(
            servicer.RegisterWorker,
            request_deserializer=supervisor_dot_v1_dot_supervisor__pb2.RegisterRequest.FromString,
            response_serializer=supervisor_dot_v1_dot_supervisor__pb2.RegisterResponse.SerializeToString,
        ),
        "StreamTasks": grpc.unary_stream_rpc_method_handler(
            servicer.StreamTasks,
            request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            response_serializer=supervisor_dot_v1_dot_supervisor__pb2.DispatchMessage.SerializeToString,
        ),
        "PushEvents": grpc.stream_unary_rpc_method_handler(
            servicer.PushEvents,
            request_deserializer=supervisor_dot_v1_dot_supervisor__pb2.EventMessage.FromString,
            response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
        ),
        "PushLogs": grpc.stream_unary_rpc_method_handler(
            servicer.PushLogs,
            request_deserializer=supervisor_dot_v1_dot_supervisor__pb2.LogMessage.FromString,
            response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
        ),
        "FetchData": grpc.unary_unary_rpc_method_handler(
            servicer.FetchData,
            request_deserializer=supervisor_dot_v1_dot_supervisor__pb2.FetchDataRequest.FromString,
            response_serializer=supervisor_dot_v1_dot_supervisor__pb2.FetchDataResponse.SerializeToString,
        ),
        "PublishData": grpc.unary_unary_rpc_method_handler(
            servicer.PublishData,
            request_deserializer=supervisor_dot_v1_dot_supervisor__pb2.PublishDataRequest.FromString,
            response_serializer=supervisor_dot_v1_dot_supervisor__pb2.PublishDataResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "supervisor.v1.Supervisor", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers(
        "supervisor.v1.Supervisor", rpc_method_handlers
    )


class Supervisor(object):
    """Supervisor streams tasks to workers and accepts events from them."""

    @staticmethod
    def RegisterWorker(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/supervisor.v1.Supervisor/RegisterWorker",
            supervisor_dot_v1_dot_supervisor__pb2.RegisterRequest.SerializeToString,
            supervisor_dot_v1_dot_supervisor__pb2.RegisterResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def StreamTasks(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/supervisor.v1.Supervisor/StreamTasks",
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            supervisor_dot_v1_dot_supervisor__pb2.DispatchMessage.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def PushEvents(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/supervisor.v1.Supervisor/PushEvents",
            supervisor_dot_v1_dot_supervisor__pb2.EventMessage.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def PushLogs(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/supervisor.v1.Supervisor/PushLogs",
            supervisor_dot_v1_dot_supervisor__pb2.LogMessage.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def FetchData(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/supervisor.v1.Supervisor/FetchData",
            supervisor_dot_v1_dot_supervisor__pb2.FetchDataRequest.SerializeToString,
            supervisor_dot_v1_dot_supervisor__pb2.FetchDataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )

    @staticmethod
    def PublishData(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/supervisor.v1.Supervisor/PublishData",
            supervisor_dot_v1_dot_supervisor__pb2.PublishDataRequest.SerializeToString,
            supervisor_dot_v1_dot_supervisor__pb2.PublishDataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
