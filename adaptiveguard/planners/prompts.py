"""Prompt contracts kept separate from provider implementations."""

PLANNER_SYSTEM_PROMPT = """You are an interactive-environment planner. Return JSON only with keys:
action (string), expected_outcome (string), confidence (HIGH, MEDIUM, or LOW).
Choose one executable action and keep the expected outcome concise."""


def planner_user_prompt(
    observation: str, goal: str, history: str, actions: list[str] | None = None
) -> str:
    candidates = "\nCandidate actions:\n" + "\n".join(f"- {a}" for a in actions) if actions else ""
    return f"Goal: {goal}\nObservation:\n{observation}\nHistory:\n{history}{candidates}"
