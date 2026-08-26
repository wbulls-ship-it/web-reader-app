# Web Reader App

A web application that extracts the main content from web pages, removes most advertisements and irrelevant elements, and presents clean, readable text.

## Current Features

- Extract article title
- Extract main article content
- Remove most advertisements and navigation elements
- Estimate reading time
- Simple Gradio web interface
- Provider-neutral text-to-speech with Kokoro as the default provider
- Automatic Chinese/English voice selection and adjustable speech speed

## Text-to-speech setup

Use Python 3.11 (3.12 is also accepted by the dependency marker) and install
`requirements.txt`. Kokoro and its current ML dependency stack did not work in the
Python 3.13 Colab runtime, so the optional Kokoro dependency is deliberately skipped
there and the app reports an actionable compatibility message. This keeps the core
article extractor importable on 3.13 instead of failing at startup. Piper remains a
registered fallback provider in the provider-neutral service, but Kokoro is the app
default.

### Kokoro performance diagnostics

Every Kokoro result now includes (and logs) separate durations for model/pipeline
loading, text preprocessing, Kokoro inference, WAV serialization, and the total
provider request. It also records `torch_device`, whether CUDA is available, the
PyTorch/CUDA versions, the selected voice, and an audio SHA-256 digest.

Run the two-voice Chinese diagnostic from the repository root:

```bash
python scripts/diagnose_kokoro.py
```

It uses one process-scoped provider (so the second run reuses the Chinese pipeline),
writes a WAV for each voice under `diagnostic-audio/`, prints the timing/device
metadata, verifies that each requested voice reaches Kokoro, and fails if the two
WAV digests are identical.

Kokoro 0.9.4 supports Python 3.11 and accepts `device="cuda"`; its automatic device
selection uses CUDA when `torch.cuda.is_available()` is true. A Colab Python 3.11
GPU runtime is therefore a practical acceleration path, provided a CUDA-enabled
PyTorch build and an assigned GPU are both present. A CUDA-capable PyTorch wheel by
itself is not evidence that a GPU is available—use the diagnostic's
`cuda_available` and `torch_device` fields. On CPU, Kokoro inference (not text
cleanup or WAV writing) is expected to dominate synthesis time.
