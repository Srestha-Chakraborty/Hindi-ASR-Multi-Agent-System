from typing import Any, Dict, List

import torch
from transformers import pipeline


def build_asr_pipeline(model_path: str = "openai/whisper-small"):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        generate_kwargs={"language": "hi", "task": "transcribe"},
    )


def transcribe_batch(asr_pipe, audio_items: List[Dict[str, Any]]) -> List[str]:
    outputs = []
    for item in audio_items:
        pred = asr_pipe(item["array"], sampling_rate=item["sampling_rate"])
        outputs.append((pred.get("text") or "").strip())
    return outputs
