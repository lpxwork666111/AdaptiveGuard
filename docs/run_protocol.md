# Run protocol

- Resolve and save the full configuration before the first episode.
- Set a fixed seed and store it in the manifest.
- Keep rule beliefs and successful-trajectory caches across an episode series.
- Reset the PFT budget and local action history at each episode boundary.
- Store every action decision, evidence vector, budget transition, and rule edit.
- Use a new run identifier when changing a task split, model, prompt, or method
  parameter.
- Never place API credentials in configuration files committed to version
  control.
