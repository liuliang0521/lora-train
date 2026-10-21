# Training project constraints

Read README.md, docs/TRAINING.md, docs/DATASET.md and docs/VALIDATION.md first.

- User-selected first route is Musubi Tuner / Qwen Edit2511 paired LoRA on RTX4060Ti16GB + 64GB RAM, with explicit memory-saving settings and GPU pilot validation. Keep SDXL as an alternative.
- Use dedicated .venv-sdxl / .venv-qwen environments. Never install training packages into ComfyUI or Jetson Python.
- Keep official trainer revisions pinned. Do not vendor their weights or copy trainer implementation into this repo.
- No models, user images, caches, run outputs or virtual environments in Git.
- Preserve raw dataset files and captions. Test fixtures are not real training or quality evidence.
- NaN/Inf, wrong architecture, unmatched LoRA pairs and smoke outputs must not be exported for deployment.
- Validate on held-out subjects before choosing checkpoints. Never silently activate Jetson candidates.
- Environment checks are not proof of real model training; report GPU/cache/backprop/quality verification separately.
