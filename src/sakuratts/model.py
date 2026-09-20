"""Model directory metadata, independent of optional inference libraries."""

from dataclasses import dataclass
import json
from pathlib import Path


FORMAT = "sakuratts-model-v1"
LEGACY_FORMAT = "sakuratts-windows-config-v1"


@dataclass(frozen=True)
class Model:
    path: Path
    manifest: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        if path.is_dir():
            path = path / "model.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format") not in (FORMAT, LEGACY_FORMAT):
            raise ValueError("Expected a sakuratts-model-v1 or sakuratts-windows-config-v1 JSON object")
        model = cls(path, manifest)
        config = model.runtime_config
        for name in ("gpt", "sovits", "frontend"):
            if not isinstance(config.get(name), str) or not config[name].strip():
                raise ValueError(f"Model requires a nonempty {name!r} package path")
        refs = config.get("references", {})
        if not isinstance(refs, dict) or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in refs.items()
        ):
            raise ValueError("Model requires named, nonempty reference package paths")
        if "default_reference" in config and config["default_reference"] not in refs:
            raise ValueError("default_reference must name a configured reference")
        for name in ("acoustic_python", "main_dictionary"):
            if name in config and (not isinstance(config[name], str) or not config[name].strip()):
                raise ValueError(f"{name} must be a nonempty path")
        if manifest["format"] == FORMAT:
            if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
                raise ValueError("Model name must be nonempty")
            if manifest.get("languages") != ["ja"]:
                raise ValueError("The public Engine currently supports Japanese models only (languages: ['ja'])")
            if manifest.get("backend", {"preferred": "cuda"}) != {"preferred": "cuda"}:
                raise ValueError("The public Engine currently supports the cuda backend only")
            for value in [config[k] for k in ("gpt", "sovits", "frontend")] + list(refs.values()):
                resource = (path.parent / value).resolve(strict=True)
                if Path(value).is_absolute() or path.parent not in resource.parents or not resource.is_dir():
                    raise ValueError("Model resources must be directories inside the model directory")
        return model

    @property
    def runtime_config(self):
        config = dict(self.manifest)
        if config.get("format") == FORMAT:
            config["format"] = LEGACY_FORMAT
            config["sovits"] = config.pop("acoustic", None)
        return config

    @property
    def name(self):
        return self.manifest.get("name", self.path.parent.name)

    @property
    def references(self):
        return tuple(self.manifest.get("references", {}))

    @property
    def default_reference(self):
        return self.manifest.get("default_reference", next(iter(self.references), None))

    def info(self):
        return {"name": self.name, "backend": "cuda", "languages": ["ja"],
                "references": list(self.references), "default_reference": self.default_reference}
