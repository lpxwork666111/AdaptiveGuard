# Environment setup

## ScienceWorld

Set `SCIENCEWORLD_ROOT` to the repository root and `SCIENCEWORLD_JAR` to the
packaged JAR. Java 11 or newer is recommended. The adapter starts a Py4J gateway
through the upstream API and shuts it down in `close()`.

Use `adaptiveguard check-env` to validate configured paths before a long run.

## VirtualHome Evolving Graph

Set `VIRTUALHOME_ROOT` to the repository root. The Evolving Graph path executes
symbolic programs entirely in Python. A manifest supplies the initial graph,
candidate actions, and goal conditions.

## VirtualHome Unity

Unity is optional for visual rendering. Download a matching upstream simulator,
set `VIRTUALHOME_SIMULATOR`, and manage its process with the upstream launcher.
The AdaptiveGuard control state remains the Evolving Graph state.
