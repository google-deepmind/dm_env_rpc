## Advice for implementers

A server implementation does not need to implement the full protocol to be
usable. When first starting, just supporting `JoinWorld`, `LeaveWorld` and
`Step` is sufficient to provide a minimum `dm_env_rpc` environment (though
clients using the provided `DmEnvAdaptor` will need access to `Reset` as well).
The world can be considered already created just by virtue of the server being
available and listening on a port. Unsupported requests should just return an
error. After this base, the following can be added:

*   `CreateWorld` and `DestroyWorld` can provide a way for clients to give
    settings and manage the lifetime of worlds.
*   Supporting sequences, where one sequence reaches its natural conclusion and
    the next begins on the next step.
*   `Reset` and `ResetWorld` can provide mechanisms to manage transitions
    between sequences and to update settings.
*   Read/Write/List properties for client code to query or set state outside of
    the normal observe/act loop.

A client implementation likely has to support the full range of features and
data types that the server it wants to interact with supports. However, it's
unnecessary to support data types or features that the server does not
implement. If the target server does not provide tensors with more than one
dimension it probably isn't worth the effort to support higher dimensional
tensors, for instance.

For Python clients, some python utility code is provided. Specifically:

*   connection.py - A utility class for managing the client-side gRPC connection
    and the error handling from server responses.
*   spec_manager.py - A utility class for managing the UID-to-name mapping, so
    the rest of the client code can use the more human readable tensor names.
*   tensor_utils.py - A few utility functions for packing and unpacking NumPy
    arrays to `dm_env_rpc` tensors.

For other languages similar utility functions will likely be needed. It is
especially recommended that client code have something similar to `SpecManager`
that turns UIDs into human readable text as soon as possible. Strings are both
less likely to be used wrong and more easily debugged.

## Reward functions

[Reward function design is difficult](https://www.alexirpan.com/2018/02/14/rl-hard.html#reward-function-design-is-difficult),
and may be the first thing a client will tweak.

For instance, a simple arcade game could expose its score function as `reward`.
However, the score function might not actually be a good measure of playing
ability if there's a series of actions which increases score without advancing
towards the actual desired goal. Alternatively, games might have an obvious
reward signal only at the end (whether the agent won or not), which might be too
sparse for a reinforcement learning agent. Constructing a robust reward function
is an area of active research and it's perfectly reasonable for an environment
to abdicate the responsibility of forming one.

For instance, constructing a reward function for the game of chess is actually
rather tricky. A simple reward at the end if an agent wins is going to make
learning difficult, as agents won't get feedback during a game if they make
mistakes or play well. Beginner chess players often use
[material values](https://en.wikipedia.org/wiki/Chess_piece_relative_value) to
evaluate a given position to decide who is winning, with queens being worth more
than rooks, etc. This might seem like a prime candidate for a reward function,
however there are
[well known shortcomings](https://en.wikipedia.org/wiki/Chess_piece_relative_value#Shortcomings_of_piece_valuation_systems)
to this simple system. For instance, using this as a reward function can blind
an agent to a move which loses material but produces a checkmate.

If a client does construct a custom reward function it may want access to data
which normally would be considered hidden and unavailable. Exposing this
information to clients may feel like cheating, however getting an AI agent to
start learning at all is often half the battle. To this end servers should
expose as much relevant information through the environment as they can to give
clients room to experiment. Weaning an agent off this information down the line
may be possible; just be sure to document which observables are considered
hidden information so a client can strip them in the future.

For client implementations, if a given server does not provide a reward or
discount observable or they aren't suitable you can build your own from other
observables. For instance, the provided `DmEnvAdaptor` has `reward` and
`discount` functions which can be overridden in derived classes.

## Nested observations example

Retrofitting an existing object-oriented codebase to be a `dm_env_rpc` server
can be difficult, as the data is likely arranged in a tree-like structure of
heterogeneous nodes. For instance, a game might have pickup and character
classes which inherit from the same base:

```
Entity {
  Vector3 position;
}

HealthPickup : Entity {
  float HealthRecoveryAmount;
  Vector4 IconColor;
}

Character : Entity {
  string Name;
}
```

It's not immediately clear how to turn this data into tensors, which can be a
difficult issue for a server implementation.

First, though, not all data needs to be sent to a joined connection. For a
health pickup, the health recovery amount might be hidden information that
agents don't generally have access to. Likewise, the IconColor might only matter
for a user interface, which an agent might not have.

After filtering unnecessary data the remainder may fit in a
"[structure of arrays](https://en.wikipedia.org/wiki/AoS_and_SoA)" format.

In our example case, we can structure the remaining data as a few different
tensors. The specs for them might look like:

```
TensorSpec {
  name = "HealthPickups.Position",
  shape = [3, -1]
  dtype = float
},

TensorSpec {
  name = "Characters.Position"
  shape = [3, -1]
  dtype = float
},

TensorSpec {
  name = "Characters.Name",
  shape = [-1]
  dtype = string
}
```

The use of a period "." character allows clients to reconstruct a hierarchy,
which can be useful in grouping tensors into logical categories.

If a structure of arrays format is infeasible, custom protocol messages can be
implemented as an alternative. This allows a more natural data representation,
but requires the client to also compile the protocol buffers and makes discovery
of metadata, such as range for numeric types, more difficult. For our toy
example, we could form the data in to this intermediate protocol buffer format:

```protobuf
message Vector3 {
  float x = 1;
  float y = 2;
  float z = 3;
}

message HealthPickup {
  Vector3 position = 1;
}

message Character {
  Vector3 position = 1;
  string name = 2;
}
```

The `TensorSpec` would then look like:

```
TensorSpec {
  name = "Characters",
  shape = [-1],
  dtype = proto
},

TensorSpec {
  name = "HealthPickups",
  shape = [-1],
  dtype = proto
},
```

### Rendering

Often renders are desirable observations for agents, whether human or machine.
In past frameworks, such as
[Atari](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning)
and
[DMLab](https://deepmind.com/blog/article/impala-scalable-distributed-deeprl-dmlab-30),
environments have been responsible for rendering. Servers implementing this
protocol will likely (but not necessarily) continue this tradition.

For performance reasons rendering should not be done unless requested by an
agent, and then only the individual renders requested. Servers can batch similar
renders if desirable.

The exact image format for a render is at the discretion of the server
implementation, but should be documented. Reasonable choices are to return an
interleaved RGB buffer, similar to a GPU render texture, or a standard image
format such as PNG or JPEG.

Human agents generally prefer high resolution images, such as the high
definition 1920x1080 format. Reinforcement agents, however, often use much
smaller resolutions, such as 96x72, for performance of both environment and
agent. It is up to the server implementation how or if it wants to expose
resolution as a setting and what resolutions it wants to support. It could be
hard coded or specified in world settings or join settings. Server implementers
are advised to consider performance impacts of whatever choice they make, as
rendering time often dominates the runtime of many environments.

### Per-element Min/Max Ranges {#per-element-ranges}

When defining actions for an environment, it's desirable to provide users with
the range of valid values that can be sent. `TensorSpec`s support this by
providing min and max range fields, which can either be defined as a single
scalar (see [broadcastable](overview.md#broadcastable)) for all elements in a
`Tensor`, or each element can have their own range defined.

For the vast majority of actions, we strongly encourage a single range only
should be defined. When users start defining per-element ranges, this is
typically indicative of the action being a combination of several, distinct
actions. By slicing the action into separate actions, there's also the benefit
of providing a more descriptive name, as well as making it easier for clients to
easily compose their own action spaces.

There are some examples where it may be desirable to not split the action. For
example, imagine an `ABSOLUTE_MOUSE_XY` action. This would have two elements
(one for the `X` and `Y` positions respectively), but would likely have
different ranges based on the width and height of the screen. Splitting the
action would mean agents could send only the `X` or `Y` action without the
other, which might not be valid.

### Documentation

Actions and observations can be self-documenting, in the sense that their
`TensorSpec`s provide information about their type, shape and bounds. However,
the exact meaning of actions and observations are not discoverable through the
protocol. In addition, `CreateWorldRequest` and `JoinWorldRequest` settings lack
even a spec. Therefore server implementers should be sure to document allowed
settings for create and join world, along with their types and shapes, the
consequences of any actions and the meaning of any observations, as well as any
properties.
