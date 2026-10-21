"""Build and run pinned upstream training commands, without shell interpolation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import ROOT, config, prepare, style_config
from .locking import training_lock
from .continuation import archive_formal_run, choose_continuation


def environment(backend, root=ROOT):
    directory = Path(root) / (".venv-" + backend)
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def commands(style, backend, run, root=ROOT, smoke=False, resume=None, pilot=False, cache_only=False):
    root, run = Path(root).resolve(), Path(run).resolve()
    source = config("backends", root)[backend]
    profile = config("profiles", root)[backend]
    selected = style_config(style, root)
    python = str(environment(backend, root))
    vendor = root / source["directory"]
    output = run / ("smoke" if smoke else "pilot" if pilot else "output")
    steps = 2 if smoke else profile.get("pilot_steps", 100) if pilot else profile.get("steps", selected["steps"])
    if sum((smoke, pilot, cache_only)) > 1 or (cache_only and backend != "qwen"):
        raise ValueError("Use one run mode; separate caching is supported for Qwen only")
    launch = [python, "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_machines", "1",
              "--num_cpu_threads_per_process", "2", "--mixed_precision", "bf16", "--dynamo_backend", "no"]
    common = ["--dataset_config", str(run / "dataset.toml"), "--mixed_precision", "bf16",
              "--network_dim", str(profile.get("rank", selected["rank"])),
              "--network_alpha", str(profile.get("alpha", selected["alpha"])),
              "--optimizer_type", profile["optimizer"],
              "--learning_rate", str(profile.get("learning_rate", selected["learning_rate"])),
              "--max_train_steps", str(steps), "--gradient_accumulation_steps", str(profile["gradient_accumulation_steps"]),
              "--gradient_checkpointing", "--sdpa", "--max_data_loader_n_workers", "0",
              "--seed", str(profile["seed"]), "--save_every_n_steps", str(profile["save_every_n_steps"]),
              "--save_state", "--output_dir", str(output),
              "--output_name", selected["id"] + "-" + backend, "--save_precision", "bf16"]
    if resume:
        common.extend(["--resume", str(Path(resume).resolve())])
    if backend == "sdxl":
        return [(vendor, launch + [str(vendor / "sdxl_train_network.py"),
                 "--pretrained_model_name_or_path", str(root / profile["checkpoint"]),
                 "--network_module", "networks.lora", "--network_train_unet_only",
                 "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs",
                 "--cache_text_encoder_outputs_to_disk", "--no_half_vae", "--save_model_as", "safetensors",
                 "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", str(0 if smoke else 100),
                 "--logging_dir", str(run / "logs"), "--log_with", "tensorboard"] + common)]
    scripts = vendor / "src/musubi_tuner"
    models = {key: str(root / profile[key]) for key in ("dit", "vae", "text_encoder")}
    memory_options = []
    if profile.get("block_swap_h2d_only"):
        memory_options += ["--block_swap_h2d_only", "--block_swap_ring_size", str(profile.get("block_swap_ring_size", 1))]
    if profile.get("gradient_checkpointing_cpu_offload"):
        memory_options += ["--gradient_checkpointing_cpu_offload"]
    cache = [
        (vendor, [python, str(scripts / "qwen_image_cache_latents.py"),
                  "--dataset_config", str(run / "dataset.toml"), "--vae", models["vae"],
                  "--model_version", "edit-2511", "--batch_size", "1", "--num_workers", "1", "--skip_existing"]),
        (vendor, [python, str(scripts / "qwen_image_cache_text_encoder_outputs.py"),
                  "--dataset_config", str(run / "dataset.toml"), "--text_encoder", models["text_encoder"],
                  "--model_version", "edit-2511", "--batch_size", "1", "--num_workers", "1", "--fp8_vl", "--skip_existing"])]
    if cache_only:
        return cache
    return cache + [
        (vendor, launch + [str(scripts / "qwen_image_train_network.py"),
                  "--dit", models["dit"], "--vae", models["vae"], "--text_encoder", models["text_encoder"],
                  "--model_version", "edit-2511", "--network_module", "networks.lora_qwen_image",
                  "--fp8_base", "--fp8_scaled", "--blocks_to_swap", str(profile["blocks_to_swap"]),
                  "--timestep_sampling", "shift", "--discrete_flow_shift", "2.2",
                  "--weighting_scheme", "none", "--lr_scheduler", "constant_with_warmup",
                  "--lr_warmup_steps", str(0 if smoke or pilot else 100),
                  "--logging_dir", str(run / ("pilot-logs" if pilot else "smoke-logs" if smoke else "logs")),
                  "--log_with", "tensorboard"] + memory_options + common)
    ]


def run_training(style, backend, root=ROOT, smoke=False, resume=None, dry_run=False, allow_small=False, pilot=False, cache_only=False, restart=False):
    if restart and (smoke or pilot or cache_only or resume or dry_run):
        raise ValueError("Restart requires formal training without resume/dry-run")
    if dry_run:
        return _run_training(style, backend, root, smoke, resume, dry_run, allow_small, pilot, cache_only, restart)
    with training_lock(root):
        return _run_training(style, backend, root, smoke, resume, dry_run, allow_small, pilot, cache_only, restart)


def _run_training(style, backend, root=ROOT, smoke=False, resume=None, dry_run=False, allow_small=False, pilot=False, cache_only=False, restart=False):
    root = Path(root)
    run = prepare(style, backend, root, allow_small)
    mode = "cache" if cache_only else "smoke" if smoke else "pilot" if pilot else "train"
    resume_step = 0
    if restart:
        archive_formal_run(run, root, backend, style)
    if mode == "train" and backend == "qwen":
        profile = config("profiles", root)[backend]
        selected = style_config(style, root)
        complete, state, resume_step = choose_continuation(
            run, selected["id"] + "-" + backend, profile.get("steps", selected["steps"]), resume)
        if complete:
            print(f"[{style}] Already complete ({resume_step} steps); skipping formal training.")
            return run
        resume = str(state) if state else None
        print(f"[{style}] Resuming from step {resume_step}: {resume}" if resume else
              f"[{style}] Starting formal training from step 0 (no saved state).")
    steps = commands(style, backend, run, root, smoke, resume, pilot, cache_only)
    plan = [{"cwd": str(cwd), "args": args} for cwd, args in steps]
    (run / (mode + "-commands.json")).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if mode == "train":
        (run / "commands.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps({"run": str(run), "commands": plan}, indent=2))
    if dry_run:
        return run
    output = run / ("smoke" if smoke else "pilot" if pilot else "output")
    if resume:
        state = Path(resume).resolve()
        if smoke or pilot or cache_only or output.resolve() not in state.parents or not state.is_dir():
            raise ValueError("Resume requires a saved state directory from this exact run's output/")
    elif not cache_only and output.exists() and any(output.glob("*.safetensors")):
        raise ValueError("Run already contains checkpoints; use --resume or archive that output before a fresh run")
    python = environment(backend, root)
    if not python.is_file():
        raise ValueError("Run scripts/setup.ps1 first: missing " + str(python))
    profile = config("profiles", root)[backend]
    for key in (("checkpoint",) if backend == "sdxl" else ("dit", "vae", "text_encoder")):
        if not (root / profile[key]).is_file():
            raise ValueError("Missing model: " + profile[key])
    vendor = steps[0][0]
    expected = config("backends", root)[backend]["revision"]
    revision = subprocess.check_output(["git", "-c", "safe.directory=" + vendor.as_posix(),
                                        "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if revision != expected:
        raise ValueError("Upstream trainer revision differs from config/backends.json")
    if backend == "qwen":
        from .backend_patches import ensure_qwen_fp8_support, ensure_qwen_resume_support
        ensure_qwen_fp8_support(vendor)
        ensure_qwen_resume_support(vendor)
        checked = subprocess.run([str(python), str(root / "scripts/check_qwen_backend.py"),
                                  "--run", str(run), "--mode", mode], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        (run / "backend-check.log").write_text(checked.stdout, encoding="utf-8")
        print(checked.stdout)
        checked.check_returncode()
    report = subprocess.check_output([str(python), "-c",
        "import json,platform,sys,torch; cuda=torch.cuda.is_available(); print(json.dumps({"
        "'python':sys.version,'executable':sys.executable,'platform':platform.platform(),"
        "'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda':cuda,"
        "'gpu':torch.cuda.get_device_name(0) if cuda else None,"
        "'vram_bytes':torch.cuda.get_device_properties(0).total_memory if cuda else None,"
        "'bf16':torch.cuda.is_bf16_supported() if cuda else False}))"], text=True)
    (run / "environment.json").write_text(report, encoding="utf-8")
    if not json.loads(report)["bf16"]:
        raise ValueError("CUDA with BF16 support is required")
    (run / "environment-freeze.txt").write_text(
        subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True), encoding="utf-8")
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               TRAINKIT_RESUME_STEP=str(resume_step))
    log_file = run / (mode + ".log")
    from .resources import ResourceMonitor
    with ResourceMonitor(run, mode) as resources, log_file.open("a", encoding="utf-8") as log:
        for index, (cwd, args) in enumerate(steps):
            resources.phase = ("cache-latents", "cache-text", "training")[index] if backend == "qwen" else "training"
            log.write("\n" + time.strftime("%Y-%m-%d %H:%M:%S") + " " + subprocess.list2cmdline(args) + "\n")
            log.flush()
            with subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError("Training/cache process failed; see " + str(log_file))
    return run
