# Release Notes

## [1.0.5]

*   Better support for string specs in `dm_env_adaptor`.
*   Improved Python type annotations.
*   Check that the server has returned the correct response in
    `dm_env_rpc.Connection` class.
*   Added `create_world` helper function to `dm_env_adaptor`.

## [1.0.4]

*   Better support for variable sized tensors.
*   Support for packing/unpacking tensors that use `Any` protobuf messages.
*   Bug fixes.

## [1.0.3]

### Added

*   Support for property descriptions.
*   New utility functions for creating a Connection instance from a server
    address.
*   DmEnvAdaptor helper functions for creating and joining worlds.
*   Additional compliance tests for resetting.
*   Support for optional DmEnvAdaptor extensions.

### Changed

*   Removed portpicker dependency, instead relying on gRPC port picking
    functionality.
*   Changed property extension API to be more ameanable to being used as an
    extension object for DmEnvAdaptor.

## [1.0.2]

*   Explicitly support nested tensors by the use of a period character in the
    `TensorSpec` name to indicate a level of nesting. Updated `dm_env` adaptor
    to flatten/unflattten actions and observations.
*   Increased minimum Python version to 3.6.
*   Moved property request/responses to its own extension. This supercedes the
    previous property requests, which have been marked as deprecated. **These
    requests will be removed in a future version of dm_env_rpc**.
*   Speed improvements for packing and un-packing byte arrays in Python.

## [1.0.1]

### Added

*   Support for per-element min/max values. This supercedes the existing scalar
    fields, which have been marked as deprecated. **These fields will be be
    removed in a future version of dm_env_rpc.**
*   Initial set of compliance tests that environment authors can use to better
    ensure their implementations adhere to the protocol specification.
*   Support for `dm_env` DiscreteArray specs.

### Changed

*   `dm_env_rpc` `EnvironmentResponse` errors in Python are now raised as a
    custom, `DmEnvRpcError` exception.

## [1.0.0]

*   Initial release.

## [1.0.0b2]

*   Updated minimum requirements for Python and protobuf.

## [1.0.0b1]

*   Initial beta release
