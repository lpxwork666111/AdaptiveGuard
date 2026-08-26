# Method specification

This document describes the implementation contract for AdaptiveGuard.

## Constraint belief

`ConstraintRule` stores a symbolic triplet, Beta parameters, an enforcement
tier, source metadata, environment scope, timestamps, update counters, and
consistency state. `StratifiedConstraintBelief` applies the lower and upper
confidence thresholds and exposes confirmed blockers and tentative matches.

## Challenge score

`PurposefulFalsificationTrials.decide` computes

```text
F = (1 - alpha / (alpha + beta)) × self_confidence × criticality
```

for every confirmed blocker. A trial requires strict passage by every blocker
and positive remaining budget. The blocker with maximum score is the unique
rule assigned the contrastive update.

## Goal criticality

`TrajectoryBuffer.criticality` combines successful-trajectory frequency,
path-position relevance, and action-template bottleneck value. It adds the
configured exploration floor and uses `0.5` before any successful trajectory is
available.

## Contrastive attribution

`ContrastiveCausalAttribution` computes the discounted normalized progression
signal, selects the environment-specific fusion schedule, thresholds the fused
value, and applies support, refutation, half-step, or typed-hypothesis updates.

`LLMHcaJudge` performs useful-hint voting, selects the longest valid hint,
requests an enhanced action, and applies a structured label. Invalid or
non-agreeing HCA data contributes zero and triggers weight renormalization.

## Consistency

Equivalent hypotheses merge into one rule. Contradictions are resolved by
confidence, behavioral cycles are broken by deprecating the lowest-confidence
edge, and oscillating beliefs are frozen after repeated flips in the configured
edit window.
