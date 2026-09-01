# Web Reader

Web Reader extracts the main text from an article URL and reads it aloud locally.
Its production TTS path is the validated bilingual **sherpa-onnx Matcha ZH+EN**
model on CPU. Kokoro remains available as an optional fallback. No cloud TTS,
account, summary service, or GPU is required.

## Architecture

The application has three deliberately separate layers:

1. `core.py` downloads and extracts articles with Trafilatura plus the existing
   Beautiful Soup fallback. TTS changes do not alter extraction behavior.
2. `tts/service.py` normalizes provider-neutral requests and selects a registered
   `TTSProvider`. `matcha`, `kokoro`, test fakes, and the legacy Piper adapter all
   implement the same interface in `tts/provider.py`.
3. `app.py` presents the URL → extract → read workflow and writes generated WAV
   bytes to a temporary directory for Gradio playback.

`MatchaTTSProvider` validates assets before loading, lazy-loads one process-scoped
`sherpa_onnx.OfflineTts` engine, detects Chinese or English automatically, passes
the selected reading speed to sherpa-onnx, and serializes mono PCM WAV output.
Matcha is registered first and is the default for both languages. Select Kokoro in
the UI only when an optional fallback is desired. Piper remains in the codebase for
compatibility but is no longer registered in the production app.

## Matcha model requirements

Install the following validated assets beneath one portable model directory:

```text
models/matcha-icefall-zh-en/
├── model-steps-3.onnx
├── vocos-16khz-univ.onnx
├── lexicon.txt
├── tokens.txt
├── espeak-ng-data/
├── phone-zh.fst
├── date-zh.fst
└── number-zh.fst
```

The model binaries are intentionally not committed. At startup the UI reports
every missing path; synthesis returns the same actionable readiness error rather
than failing with an opaque ONNX error.

The default root is relative to the working directory and contains no Colab-only
path. Override it on Linux/macOS:

```bash
export MATCHA_MODEL_DIR=/absolute/path/to/matcha-icefall-zh-en
export MATCHA_NUM_THREADS=2
```

Or in Windows PowerShell:

```powershell
$env:MATCHA_MODEL_DIR = 'C:\WebReader\models\matcha-icefall-zh-en'
$env:MATCHA_NUM_THREADS = '2'
```

For custom layouts, each asset can be set independently with
`MATCHA_ACOUSTIC_MODEL`, `MATCHA_VOCODER`, `MATCHA_LEXICON`, `MATCHA_TOKENS`,
`MATCHA_ESPEAK_DATA`, `MATCHA_PHONE_FST`, `MATCHA_DATE_FST`, and
`MATCHA_NUMBER_FST`.

## Local CPU setup

Python 3.11 or 3.12 is recommended (Kokoro is skipped by dependency markers on
Python 3.13). Create an isolated environment and install dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned `sherpa-onnx==1.13.6` package and the provider's `provider="cpu"`
configuration ensure CPU-only inference. `MATCHA_NUM_THREADS` defaults to `2`,
matching the validated two-core benchmark environment (approximately 0.104 English
RTF and 0.110 Chinese RTF). A GPU runtime is neither selected nor needed.

After placing the assets, run:

```bash
python app.py
```

Open the local Gradio address, enter an HTTP(S) article URL, select **Extract
Article**, then **Read Aloud**. The default provider handles Chinese and English
without a voice switch. Reading speed ranges from 0.5× through 2.0× and the result
appears in the audio player.

## Tests

```bash
python -m unittest discover -v
```

The suite covers extraction preservation, provider selection, automatic Chinese
and English Matcha routing, missing assets, speed propagation, WAV validity, and
the optional/legacy provider adapters. Tests use a small injected engine and do not
require downloading model binaries.

## Windows-local MVP notes

The application and path configuration are Windows-portable, but a Windows MVP
still needs the model archive to be distributed/installed at the configured path
and a clean-machine validation of the sherpa-onnx 1.13.6 wheel, espeak data lookup,
firewall/browser launch behavior, and end-to-end audio latency. Those deployment
checks are the remaining work; there is no known application-architecture blocker.
