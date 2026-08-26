# Architecture

`AdaptiveGuardController` owns the online control loop while all model and
environment operations are injected through typed interfaces.

```text
observation + goal + history
            │
            ▼
 Planner(action, expected outcome, confidence)
            │
            ▼
 SCB confirmed blockers ──► PFT ──► ALLOW / TRIAL / BLOCK
                                      │
                                      ▼
                                 environment
                                      │
                         observation, error, score
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                        PRJ          SPS          HCA
                         └────────────┼────────────┘
                                      ▼
                              CCA fusion and edit
                                      │
                                      ▼
                         consistency + tier refresh
```

The belief graph is persistent across episodes. The PFT budget resets per
episode, and successful trajectories extend the criticality cache.

All step data is represented by immutable typed records where possible. Mutable
state is confined to the rule graph, budget, trajectory cache, simulator, and
controller history.
