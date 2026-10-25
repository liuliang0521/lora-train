"""Installed by test-loras; verify every trained LoRA layer before sampling."""
import torch
import comfy.lora
import comfy.utils
import folder_paths


class TrainingValidationLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 0.85, "min": 0, "max": 2})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "training/validation"

    def apply(self, model, lora_name, strength_model):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        weights = comfy.utils.load_torch_file(path, safe_load=True)
        for key, value in weights.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("NaN/Inf in LoRA: " + key)
        mapping = comfy.lora.model_lora_keys_unet(model.model, {})
        patches = comfy.lora.load_lora(weights, mapping)
        expected = sum(k.endswith(".lora_down.weight") for k in weights)
        if not expected or len(patches) != expected:
            raise ValueError(f"LoRA/base model mismatch: {len(patches)}/{expected} layers")
        result = model.clone()
        accepted = result.add_patches(patches, strength_model)
        if len(accepted) != expected:
            raise ValueError(f"Only {len(accepted)}/{expected} LoRA layers were applied")
        return (result,)


NODE_CLASS_MAPPINGS = {"TrainingValidationLora": TrainingValidationLora}
