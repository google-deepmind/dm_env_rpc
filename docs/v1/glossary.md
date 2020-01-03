## Glossary

### Server

A server is a process which implements the dm_env_rpc `Environment` gRPC
service. A server can host one or more worlds and allow one or more client
connections.

### Client

A client connects to a server. It can request worlds to be created and join them
to become agents. It can also query properties and reset worlds without being an
agent.

### Agent

A client which has joined a world. In a game sense this is a "player". It has
the ability to send actions and receive observations. This could be a human or a
reinforcement learning agent.

### World

A system or simulation in which one or more agents interact.

### Environment

An agent's view of a world. In limited information situations, the environment
may expose only part of the world state to an agent that corresponds to
information that agent is allowed by the simulation to have. Agents communicate
directly with an environment, and the environment communicates with the world to
synchronize with it.

### Sequence

A series of discrete states, where one state is correlated to previous and
subsequent states, possibly ending in a terminal state, and usually modified by
agent actions. In the simplest case, playing an entire game until one player is
declared the winner is one sequence. Also sometimes called an "episode" in
reinforcement learning contexts.
