"""Implemented language capabilities; importing this module loads no models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    name: str
    modes: tuple[str, ...]
    terminal: str
    resource_format: str


JAPANESE = LanguageProfile(
    "ja", "Japanese", ("ja", "all_ja"), "。", "sakuratts-japanese-frontend-resources-v1",
)
LANGUAGE_PROFILES = {JAPANESE.code: JAPANESE}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_PROFILES)
SUPPORTED_LANGUAGE_MODES = tuple(mode for profile in LANGUAGE_PROFILES.values() for mode in profile.modes)
DEFAULT_LANGUAGE = JAPANESE.code


def language_profile(mode):
    for profile in LANGUAGE_PROFILES.values():
        if mode in profile.modes:
            return profile
    supported = ", ".join(SUPPORTED_LANGUAGE_MODES)
    names = ", ".join(profile.name for profile in LANGUAGE_PROFILES.values())
    raise ValueError(f"Unsupported language mode {mode!r}; available {names} modes: {supported}")
