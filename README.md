# AdaptiveGuard

AdaptiveGuard is an implementation of adaptive guarded planning for
LLM agents in dynamic interactive environments. It represents admissibility
constraints as falsifiable probabilistic beliefs and updates those beliefs from
environment-grounded evidence.

The implementation contains three connected modules:

- **SCB — Stratified Constraint Belief:** stores every symbolic rule with a
  Beta belief and a tentative, confirmed, or deprecated enforcement tier.
- **PFT — Purposeful Falsification Trials:** selectively executes a blocked
  action when rule uncertainty, planner confidence, goal criticality, and the
  bounded trial budget jointly permit a challenge.
- **CCA — Contrastive Causal Attribution:** combines process judgment, score
  progression, and hindsight-guided attribution to strengthen, weaken, or add
  typed constraints.

PFT is deterministic and reuses confidence emitted by the planner's normal
forward pass. It does not add a model call to the action-decision path.

## Repository layout

```text
adaptiveguard/
├── adaptiveguard/
│   ├── core/          # rules, beliefs, trajectories, controller, audit types
│   ├── modules/       # SCB, PFT, CCA, budget, consistency
│   ├── planners/      # planner protocol, mock and OpenAI-compatible planners
│   ├── judges/        # PRJ, HCA, static prompt audit
│   ├── environments/  # ScienceWorld, VirtualHome and local mock adapters
│   ├── bootstrap/     # demonstration, engine and oracle rule construction
│   ├── evaluation/    # run aggregation and rule-graph diagnostics
│   ├── io/            # YAML, JSON/JSONL, manifests and logging
│   └── cli/           # command-line entry points
├── configs/           # method, environment and model settings
├── docs/              # architecture and interface specifications
├── examples/          # executable examples and a VirtualHome task manifest
├── scripts/           # convenience launchers
└── tests/             # unit and integration tests
```

## Installation

Python 3.10 or newer is required. ScienceWorld additionally requires a Java
runtime. The supplied Conda file installs both:

```bash
cd /path/to/adaptiveguard
conda env create -f environment.yml
conda activate adaptiveguard
python -m pip install -e '.[dev,environments]'
```

Alternatively, in an existing Python environment:

```bash
python -m pip install -e '.[dev,environments]'
```

Copy `.env.example` to a private shell configuration or export the variables
directly. The program never loads `.env` implicitly, which keeps secret
handling explicit.

## Local quick start

The default configuration uses deterministic local components and needs no
simulator or API credential:

```bash
adaptiveguard run --config configs/default.yaml --run-id quickstart
python examples/mock_episode.py
```

Inspect the current rule graph with:

```bash
adaptiveguard inspect-rules --rules outputs/quickstart/rules.json
```

## ScienceWorld

Point AdaptiveGuard at a local ScienceWorld checkout:

```bash
export SCIENCEWORLD_ROOT=/path/to/ScienceWorld
export SCIENCEWORLD_JAR="$SCIENCEWORLD_ROOT/scienceworld/scienceworld.jar"
adaptiveguard check-env
adaptiveguard run --config configs/scienceworld.yaml
```

The adapter uses `ScienceWorldEnv`, including task loading, variations,
simplifications, possible actions, score deltas, validity feedback, step limits,
and clean JVM shutdown. Change `task_name`, `variation`, and `simplifications`
under the configuration's `environment` section.

## VirtualHome

The default adapter uses the pure-Python Evolving Graph simulator, so Unity is
not required for symbolic action execution:

```bash
export VIRTUALHOME_ROOT=/path/to/virtualhome
adaptiveguard run --config configs/virtualhome.yaml
```

A VirtualHome task manifest contains:

- an initial graph or a path to an initial-graph JSON file;
- name equivalences;
- executable candidate action strings;
- node-state and edge goal conditions;
- an optional character index.

See `examples/data/virtualhome_task.json` for the complete schema. Candidate
actions use VirtualHome's native notation, for example:

```text
[SWITCHON] <lamp> (3)
```

Unity rendering remains optional and independent of the Evolving Graph control
path. Set `VIRTUALHOME_SIMULATOR` when a Unity executable is managed by an
external launcher.

Install the Unity communication dependencies and select its configuration when
Unity execution is required:

```bash
python -m pip install -e '.[unity]'
adaptiveguard run --config configs/virtualhome_unity.yaml
```

## OpenAI-compatible model configuration

The built-in client uses the standard chat-completions JSON contract and only
depends on the Python standard library:

```bash
export ADAPTIVEGUARD_LLM_BASE_URL=https://api.openai.com/v1
export ADAPTIVEGUARD_LLM_API_KEY=your-key
export ADAPTIVEGUARD_LLM_MODEL=your-model
```

Provider settings support deterministic temperature, request timeout, retry
count, and maximum output tokens. Planner output is schema-checked and must be:

```json
{
  "action": "turn on heater",
  "expected_outcome": "the heater starts warming the container",
  "confidence": "HIGH"
}
```

PRJ and HCA use separate prompt contracts. Judgment prompts are statically
audited before transmission and cannot contain a serialized constraint graph.

## Initial rule graphs

Create rules from a JSONL demonstration file:

```bash
adaptiveguard bootstrap \
  --input data/demonstrations.jsonl \
  --output outputs/rules/initial.json \
  --environment scienceworld
```

Each line may contain a top-level symbolic triplet:

```json
{"head":"heater","relation":"requires_previous","tail":"charge heater","metadata":{"action_template":"turn on"}}
```

or the same fields nested under `rule`. Repeated support is aggregated into the
initial Beta belief. Engine and oracle helpers are available from
`adaptiveguard.bootstrap.engine_rules` and
`adaptiveguard.bootstrap.oracle_rules`.

## Rule representation

Serialized rules use a stable symbolic identifier and retain their full audit
history:

```json
{
  "rule_id": "stable-hash",
  "head": "heater",
  "relation": "requires_previous",
  "tail": "charge heater",
  "alpha": 4.0,
  "beta": 1.0,
  "tier": "confirmed",
  "source": "demonstration",
  "environment": "scienceworld",
  "metadata": {}
}
```

The configured thresholds are applied periodically. Deprecated rules remain in
storage for audit but do not block actions. Tentative rules participate in
attribution without acting as hard blockers.

## Run artifacts

Every run creates an isolated directory containing:

```text
config.json       exact resolved configuration
manifest.json     timestamp, seed, Python and platform information
episodes.jsonl    step-level planner, PFT, environment and CCA records
summary.json      compact run summary
rules.json        final versioned rule graph
```

JSON rule writes are atomic. Episode records use append-only JSONL so a partial
run remains inspectable.

## Extending AdaptiveGuard

Implement `adaptiveguard.planners.base.Planner` for a custom action planner.
Implement `ProcessRewardJudge` and `HcaJudge` for custom environment-grounded
judges. New environments implement `InteractiveEnvironment.reset`, `step`, and
optionally `available_actions` and `close`.

The complete control flow is in
`adaptiveguard.core.controller.AdaptiveGuardController`; the environment,
planner, judges, SCB, PFT, and CCA are passed as explicit dependencies.

## Development checks

```bash
make format
make lint
make typecheck
make test
```

Integration tests that need external simulators use availability checks. The
mock integration path always runs and exercises the full closed loop.

## Citation and upstream software

Citation metadata is provided in `CITATION.cff`. ScienceWorld and VirtualHome
remain separate upstream projects with their own licenses; AdaptiveGuard imports
their public Python interfaces and does not vendor either simulator.
