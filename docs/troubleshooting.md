# Troubleshooting

## ScienceWorld import fails

Set `SCIENCEWORLD_ROOT` to the directory containing the `scienceworld` package,
then install `py4j` or install the upstream checkout in editable mode.

## Java gateway does not start

Confirm `java -version`, verify `SCIENCEWORLD_JAR`, and ensure no security policy
blocks local callback ports.

## VirtualHome action is not executable

Use native syntax such as `[OPEN] <fridge> (42)`. Confirm object identifiers,
character proximity, object properties, and initial graph edges.

## Model response cannot be parsed

The endpoint must honor JSON-object mode and return all required planner fields.
Inspect the stored raw response, reduce temperature, and keep prompt templates
unchanged while diagnosing schema failures.

## All challenged actions are blocked

Inspect rule confidence, planner confidence, criticality, `tau`, and remaining
budget. A trial is allowed only when every confirmed blocker independently
passes the strict gate.
