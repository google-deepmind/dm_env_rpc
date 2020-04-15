# Extensions

`dm_env_rpc` Extensions provide a way for users to send custom messages over the
same bi-directional gRPC stream as the standard messages. This can be useful for
debugging or manipulating the simulation in a way that isn't appropriate through
the typical `dm_env_rpc` protocol.

Extensions must complement the existing `dm_env_rpc` protocol, so that agents
are able to send conventional `dm_env_rpc` messages, interspersed with extension
requests.

To send an extension message, you must pack your custom message into an `Any`
proto, assigning to the `extension` field in `EnvironmentRequest`. For example:

```python
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from google.protobuf import any_pb2
from google.protobuf import struct_pb2

def send_custom_message(connection: dm_env_rpc_connection.Connection):
  """Send/receive custom Struct message through a dm_env_rpc connection."""
  my_message = struct_pb2.Struct(
          fields={'foo': struct_pb2.Value(string_value='bar')})

  packed_message = any_pb2.Any()
  packed_message.Pack(my_message)

  response = struct_pb2.Struct()
  connection.send(packed_message).Unpack(response)
```

If the simulation can respond to such a message, it must send a response using
the corresponding `extension` field in `EnvironmentResponse`. The server must
send an [error](../overview.md#errors) if it doesn't recognise the request.

## Common Extensions

The following commonly used extensions are provided with `dm_env_rpc`:

*   [Properties](properties.md)

## Recommendations

### Should you use an extension?

Although extensions can be powerful, if you expect an extension message to be
sent by the client every step, consider making it a proper action or observable,
even if it's intended as metadata. This better ensures that all mutable actions
can be executed at well-ordered times.

### Creating one parent request/response for your extension

If your extension has more than a couple of requests, consider creating a single
parent request/response message that you can add/remove messages from. This
simplifies the server code by only having to unpack a single request, and makes
it easier to compartmentalize each extension. For example:

```proto
message AwesomeRequest {
  string foo = 1;
}

message AwesomeResponse {}

message AnotherAwesomeRequest {
  string bar = 1;
}

message AnotherAwesomeResponse {}

message MyExtensionRequest {
  oneof payload {
    AwesomeRequest awesome = 1;
    AnotherAwesomeRequest another_awesome = 2;
  }
}

message MyExtensionResponse {
  oneof payload {
    AwesomeResponse awesome = 1;
    AnotherAwesomeResponse another_awesome = 2;
  }
}
```

## Alternatives

An alternative to `dm_env_rpc` extension messages is to register a separate gRPC
service. This has the benefit of being able to use gRPC's other
[service methods](https://grpc.io/docs/guides/concepts/#service-definition).
However, if you need messages to be sent at a particular point (e.g. after a
specific `StepRequest`), synchronizing these disparate services will add
additional complexity to the server.
