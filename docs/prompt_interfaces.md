# Prompt interfaces

## Planner

Input: observation, goal, action history, and optional candidate actions.

Output: JSON fields `action`, `expected_outcome`, and `confidence`. Confidence
must be `HIGH`, `MEDIUM`, or `LOW` and is mapped to `0.9`, `0.6`, or `0.3`.

## Process reward judge

Input: executed action, next observation, environment feedback, and normalized
score change. Each vote is `-1`, `0`, or `+1`. CCA retains both the modal label
and signed vote mean.

## Hindsight-guided attribution

The hint phase compares predicted and observed outcomes. The longest useful
hint is added to the enhanced-action query. The final rubric maps the original
and enhanced actions to one of six typed labels.

## Leakage audit

PRJ, hint, and structured-rubric payloads are checked before transmission. A
payload containing serialized rule text or belief-graph structure raises
`PromptLeakageError`. Only post-judgment hypothesis construction may use the
target graph type.
