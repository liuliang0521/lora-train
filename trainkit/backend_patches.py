"""Small guarded compatibility patches for the pinned Musubi checkout."""

ORIGINAL_NORM = '                    if module.__class__.__name__ in ["Qwen2RMSNorm"]:'
FP8_NORM = '                    if module.__class__.__name__ in ["Qwen2RMSNorm", "Qwen2_5_VLRMSNorm"]:'
ORIGINAL_DECODER = '''                    if module.__class__.__name__ in ["Qwen2_5_VLDecoderLayer"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = decoder_forward_hook(module)'''
FP8_DECODER = '''                    if module.__class__.__name__ in ["Qwen2_5_VLDecoderLayer"]:
                        # Transformers 5 uses the native Tensor-returning decoder.
                        if "past_key_values" not in inspect.signature(module.forward).parameters:
                            module.forward = decoder_forward_hook(module)'''


def ensure_qwen_fp8_support(vendor):
    """Persist the RMSNorm/decoder fixes without tracking vendor sources."""
    source = vendor / "src/musubi_tuner/qwen_image/qwen_image_utils.py"
    original = source.read_text(encoding="utf-8")
    patched = original
    if FP8_NORM not in patched:
        if patched.count(ORIGINAL_NORM) != 1:
            raise ValueError("Cannot apply Qwen FP8 RMSNorm compatibility patch: " + str(source))
        patched = patched.replace(ORIGINAL_NORM, FP8_NORM, 1)
    decoder_guard = 'if "past_key_values" not in inspect.signature(module.forward).parameters:'
    if decoder_guard not in patched:
        if patched.count(ORIGINAL_DECODER) != 1:
            raise ValueError("Cannot apply Qwen FP8 decoder compatibility patch: " + str(source))
        patched = patched.replace(ORIGINAL_DECODER, FP8_DECODER, 1)
    if "import inspect\n" not in patched:
        patched = "import inspect\n" + patched
    if patched != original:
        compile(patched, str(source), "exec")
        source.write_text(patched, encoding="utf-8")

ORIGINAL_PROGRESS = '''        # TODO skip until initial step
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

        epoch_to_start = 0
        global_step = 0'''

RESUMED_PROGRESS = '''        # trainkit: resume the optimizer-step counter as well as model/optimizer.
        global_step = int(os.environ.get("TRAINKIT_RESUME_STEP", "0")) if args.resume else 0
        if not 0 <= global_step <= args.max_train_steps:
            raise ValueError("Invalid TRAINKIT_RESUME_STEP")
        progress_bar = tqdm(range(args.max_train_steps), initial=global_step, smoothing=0,
                            disable=not accelerator.is_local_main_process, desc="steps")
        epoch_to_start = global_step // num_update_steps_per_epoch
        resume_batches = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps'''

ORIGINAL_LOOP = '''        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print'''
RESUMED_LOOP = '''        for epoch in range(epoch_to_start, num_train_epochs):
            if global_step >= args.max_train_steps:
                break
            accelerator.print'''

ORIGINAL_BATCHES = '''            for step, batch in enumerate(train_dataloader):
                # torch.compiler.cudagraph_mark_step_begin()'''
RESUMED_BATCHES = '''            epoch_dataloader = train_dataloader
            if epoch == epoch_to_start and resume_batches:
                epoch_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batches)
            for step, batch in enumerate(epoch_dataloader):
                # torch.compiler.cudagraph_mark_step_begin()'''


def ensure_qwen_resume_support(vendor):
    """Fail closed if upstream changed; preserve all unrelated local patches."""
    source = vendor / "src/musubi_tuner/training/trainer_base.py"
    original = source.read_text(encoding="utf-8")
    patched = original
    for before, after in ((ORIGINAL_PROGRESS, RESUMED_PROGRESS),
                          (ORIGINAL_LOOP, RESUMED_LOOP), (ORIGINAL_BATCHES, RESUMED_BATCHES)):
        if after in patched:
            continue
        if patched.count(before) != 1:
            raise ValueError("Cannot apply Musubi resume compatibility patch: " + str(source))
        patched = patched.replace(before, after, 1)
    if patched != original:
        compile(patched, str(source), "exec")
        source.write_text(patched, encoding="utf-8")
