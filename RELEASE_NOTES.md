# Release Notes

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
