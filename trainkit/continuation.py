"""Choose resumable formal checkpoints and archive only formal training artifacts."""
from pathlib import Path
import time
import uuid
import zipfile


def checkpoint_step(path):
    """Read metadata without loading weights; reject interrupted/truncated saves."""
    from safetensors import safe_open
    path = Path(path)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            if not list(stream.keys()):
                return None
            value = int((stream.metadata() or {}).get("ss_steps", "-1"))
            return value if value >= 0 else None
    except (OSError, ValueError, TypeError):
        return None
    except Exception as error:
        # safetensors exposes its parse exception separately from ValueError.
        from safetensors import SafetensorError
        if isinstance(error, SafetensorError):
            return None
        raise


def valid_state(state, output):
    """Only local, complete single-process Accelerate states are candidates."""
    state, output = Path(state), Path(output).resolve()
    if state.is_symlink() or not state.is_dir() or state.resolve().parent != output:
        return False
    model = state / "model.safetensors"
    if model.is_symlink():
        return False
    try:
        from safetensors import safe_open
        with safe_open(str(model), framework="pt", device="cpu") as stream:
            if not list(stream.keys()):
                return False
        for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
            path = state / name
            if path.is_symlink() or not path.is_file():
                return False
            # torch.save uses ZIP; its central directory is written last.
            # Inspect that directory without unpickling optimizer/RNG objects.
            with zipfile.ZipFile(path) as archive:
                if not archive.namelist():
                    return False
    except (OSError, ValueError, zipfile.BadZipFile):
        return False
    except Exception as error:
        from safetensors import SafetensorError
        if isinstance(error, SafetensorError):
            return False
        raise
    return True


def choose_continuation(run, output_name, total_steps, explicit=None):
    """Return (complete, state, step), scoped to this exact data/profile run."""
    output = Path(run) / "output"
    if output.is_symlink():
        raise ValueError("Refusing linked output directory: " + str(output))
    if explicit:
        state = Path(explicit).resolve()
        if not state.name.endswith("-state") or not valid_state(state, output):
            raise ValueError("Resume requires a complete state directory in this exact run's output/")
        checkpoint = state.with_name(state.name[:-len("-state")] + ".safetensors")
        step = checkpoint_step(checkpoint)
        if step is None or not 0 < step <= total_steps:
            raise ValueError("Resume checkpoint has missing/invalid ss_steps metadata")
        return False, state, step
    final = output / (output_name + ".safetensors")
    if checkpoint_step(final) == total_steps:
        return True, None, total_steps
    candidates = []
    for state in output.glob(output_name + "*-state"):
        checkpoint = state.with_name(state.name[:-len("-state")] + ".safetensors")
        step = checkpoint_step(checkpoint)
        if step is not None and 0 < step <= total_steps and valid_state(state, output):
            candidates.append((step, state.name, state))
    if candidates:
        step, _, state = max(candidates)
        return False, state, step
    if output.exists() and any(output.iterdir()):
        raise ValueError("Formal output exists but no complete resumable state was found; "
                         "use -Restart (CLI: --restart) to archive it and start over")
    return False, None, 0


def archive_formal_run(run, root, backend, style):
    """Recoverably clear the current fingerprint, never raw data/cache/trial runs."""
    run, root = Path(run), Path(root).resolve()
    expected_parent = root / "runs" / backend / style
    if run.is_symlink() or run.resolve().parent != expected_parent:
        raise ValueError("Refusing to archive outside this style's run directory")
    names = ("output", "logs", "train.log", "train-resources.json", "train-resources.jsonl",
             "train-commands.json", "commands.json")
    targets = [run / name for name in names if (run / name).exists() or (run / name).is_symlink()]
    archive_root = run / "archive"
    # Validate every source and destination before moving anything.
    if archive_root.is_symlink() or archive_root.resolve().parent != run.resolve():
        raise ValueError("Refusing linked archive directory")
    for target in targets:
        if target.is_symlink() or target.resolve().parent != run.resolve():
            raise ValueError("Refusing linked formal training artifact: " + str(target))
    if not targets:
        return None
    destination = archive_root / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True)
    for target in targets:
        target.rename(destination / target.name)
    print("Archived previous formal training results to " + str(destination))
    return destination
