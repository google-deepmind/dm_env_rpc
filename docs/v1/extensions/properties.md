## Properties

Often side-channel data is useful for debugging or manipulating the simulation
in some way which isn’t appropriate for an agent’s interface. The property
system provides this capability, allowing both the reading, writing and
discovery of properties. Properties are implemented as an extension to the core
protocol (see [extensions](index.md) for more details).

Properties queried before a `JoinWorldRequest` will correspond to universal
properties that apply for all worlds. Properties queried after a
JoinWorldRequest can add a layer of world-specific properties.

For writing properties, it’s up to the server to determine if/when any
modification should take place (e.g. changing the world seed might not take
place until the next sequence).

For reading properties, the exact timing of the observation is up to the world.
It may occur at the previous step's time, or it may occur at some intermediate
time.

Properties can be laid out in a tree-like structure, as long as each node in the
tree has a unique key, by having parent nodes be listable (the is_listable field
in the Property message).

Although properties can be powerful, if you expect a property to be read or
written to every step by a normally functioning agent, it may be preferable to
make it a proper action or observation, even if it’s intended to be metadata.
For instance, a score which provides hidden information could be done as a
property, but it might be preferable to do it as an observation. This ensures
observations and actions all occur at well-ordered times.

### Read Property

```proto
package dm_env_rpc.v1.extensions.properties;

message ReadPropertyRequest {
  string key = 1;
}

message ReadPropertyResponse {
  dm_env_rpc.v1.Tensor value = 1;
}
```

Returns the current value for the property with the provided `key`.

### Write Property

```proto
package dm_env_rpc.v1.extensions.properties;

message WritePropertyRequest {
  string key = 1;
  dm_env_rpc.v1.Tensor value = 2;
}

message WritePropertyResponse {}
```

Set the property referened by the `key` field to the value of the provided
`Tensor`.

### List Property

```proto
package dm_env_rpc.v1.extensions.properties;

message ListPropertyRequest {
  // Key to list property for. Empty string is root level.
  string key = 1;
}

message ListPropertyResponse {
  message PropertySpec {
    // Required: TensorSpec name field for key value.
    dm_env_rpc.v1.TensorSpec spec = 1;

    bool is_readable = 2;
    bool is_writable = 3;
    bool is_listable = 4;
  }

  repeated PropertySpec values = 1;
}
```

Returns an array of `PropertySpec` values for each property residing under the
provided `key`. If the `key` is empty, properties registered at the root level
are returned. If the `key` is not empty and the property is not listable an
error is returned.

Each `PropertySpec` must return a `TensorSpec` with its name field set to the
property's key. For readable and writable properties, the type and shape of the
property must also be set. Properties which are only listable must have the
default value for type (`dm_env_rpc.v1.DataType.INVALID_DATA_TYPE`) and shape
(an empty array).
