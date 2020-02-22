# `dm_env_rpc`: A networking protocol for agent-environment communication.

`dm_env_rpc` is a remote procedure call (RPC) protocol for communicating between
machine learning agents and environments. It uses [gRPC](http://www.grpc.io) as
the underlying communication framework, specifically its
[bidirectional streaming](http://grpc.io/docs/guides/concepts/#bidirectional-streaming-rpc)
RPC variant.

This package also contains an implementation of
[`dm_env`](http://www.github.com/deepmind/dm_env), a Python interface for
interacting with such environments.

Please see the documentation for more detailed information on the semantics of
the protocol and how to use it. The examples sub-directory also provides
examples of RL environments implemented using the `dm_env_rpc` protocol.

## Intended audience

Games can make for interesting AI research platforms, for example as
reinforcement learning (RL) environments. However, exposing a game as an RL
environment can be a subtle, fraught process. We aim to provide a protocol that
allows agents and environments to communicate in a standardized way, without
specialized knowledge about how the other side works. Game developers can expose
their games as environments with minimal domain knowledge and researchers can
test their agents on a large library of different games.

This protocol also removes the need for agents and environments to run in the
same process or even on the same machine, allowing agents and environments to
have very different technology stacks and requirements.

## Documentation

*   [Protocol overview](docs/v1/overview.md)
*   [Protocol reference](docs/v1/reference.md)
*   [Appendix](docs/v1/appendix.md)
*   [Glossary](docs/v1/glossary.md)

## Installation

Note: You may optionally wish to create a
[Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to
prevent conflicts with your system's Python environment.

`dm_env_rpc` can be installed from [PyPi](https://pypi.org/project/dm-env-rpc/)
using `pip`:

```bash
$ pip install dm-env-rpc
```

To also install the dependencies for the `examples/`, install with:

```bash
$ pip install dm-env-rpc[examples]
```

Alternatively, you can install `dm_env_rpc` by cloning a local copy of our
GitHub repository:

```bash
$ git clone --recursive https://github.com/deepmind/dm_env_rpc.git
$ pip install ./dm_env_rpc
```

## Citing `dm_env_rpc`

To cite this repository:

```bibtex
@misc{dm_env_rpc2019,
  author = {Tom Ward and Jay Lemmon},
  title = {dm\_env\_rpc: A networking protocol for agent-environment communication},
  url = {http://github.com/deepmind/dm_env_rpc},
  year = {2019},
}
```

## Notice

This is not an officially supported Google product
