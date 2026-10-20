# Training project constraints

Read README.md, docs/TRAINING.md, docs/DATASET.md and docs/VALIDATION.md first.

- Default route is SDXL style LoRA on RTX4060Ti16GB. Qwen Edit2511 is an optional paired-data experiment.
- Use dedicated .venv-sdxl / .venv-qwen environments. Never install training packages into ComfyUI or Jetson Python.
- Keep official trainer revisions pinned. Do not vendor their weights or copy trainer implementation into this repo.
- No models, user images, caches, run outputs or virtual environments in Git.
- Preserve raw dataset files and captions. Test fixtures are not real training or quality evidence.
- NaN/Inf, wrong architecture, unmatched LoRA pairs and smoke outputs must not be exported for deployment.
- Validate on held-out subjects before choosing checkpoints. Never silently activate Jetson candidates.
- Environment checks are not proof of real model training; report GPU/cache/backprop/quality verification separately.
