"""CUDA runtime construction and its backend-specific options."""


def create_runtime(model, *, experimental=None, load_references=True):
    options = dict(experimental or {})
    allowed = {"policy", "use_graph", "capacity", "gpt_precision", "gpt_attention",
               "gpt_attention_chunk_size", "gpt_prefill_query_chunk_size", "allow_experimental_acoustic_fp16",
               "acoustic_arena_shrink", "acoustic_chunk_frames", "acoustic_session_policy"}
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError("Unknown experimental options: " + ", ".join(sorted(unknown)))
    if not load_references:
        options["load_references"] = False
    from .engine import NVIDIAEngine
    return NVIDIAEngine(model, **options)
