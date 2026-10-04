"""Read upstream configuration without loading inference resources."""

from pathlib import Path


def read_inference_configuration(model=None, *, tts_config=None):
    """Read service settings without constructing frontends or GPU resources."""
    import json
    settings = {}
    if tts_config:
        path = Path(tts_config)
        if path.suffix.lower() == ".json":
            config = json.loads(path.read_text(encoding="utf-8"))
        else:
            import yaml
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("TTS configuration must be a mapping")
        if "format" in config:
            model = path
        else:
            settings = dict(config.get("sakuratts", {}))
            custom = config.get("custom", config)
            if "device" in custom:
                settings.setdefault("backend", custom["device"])
            if custom.get("is_half", False):
                raise NotImplementedError("Official is_half mode is not yet supported; use is_half: false")
            if custom.get("version", "v2ProPlus") not in ("v2Pro", "v2ProPlus"):
                raise NotImplementedError("The native service currently supports v2Pro and v2ProPlus")
            model = model or settings.get("model")
            for key, field in (("gpt", "t2s_weights_path"), ("sovits", "vits_weights_path")):
                if custom.get(field):
                    settings[key + "_checkpoint"] = custom[field]
            if custom.get("cnhuhbert_base_path"):
                settings["cnhubert"] = custom["cnhuhbert_base_path"]
    from ..runtime.portable import preparation_settings
    if "runtime_options" in settings and not isinstance(settings["runtime_options"], dict):
        raise ValueError("sakuratts.runtime_options must be a mapping")
    return model, preparation_settings(settings)
