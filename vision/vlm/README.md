# vision/vlm

Vision-Language Model (VLM) module. Turns a camera frame plus a natural-language
prompt into a textual answer using an LLM with vision capabilities. Implements
the `VisionClient` interface so it can run standalone (`__main__`) or be composed
by `VisionPipeline` (on a throttled cadence for scene captioning).

## Files

- **`base.py`** — `VisionLanguageModel` ABC with
  `analyze(frame, prompt, history=None) -> VLMResponse`, plus `VLMQuery` /
  `VLMResponse` dataclasses.
- **`factory.py`** — `create_vlm(name, **kwargs)` builder.
- **`openai.py`** — `OpenAIVisionLM`, the LangChain + OpenAI implementation
  (`gpt-4o-mini` by default). Encodes the frame to a JPEG data-URL and sends it
  alongside the prompt. Reads `OPENAI_API_KEY` from `.env` (via `init_env`).
- **`client.py`** — `VLMClient(VisionClient)`, the single entry point. Its
  `run` performs a default scene caption; use `analyze(frame, prompt)` for
  custom questions.
- **`__main__.py`** — standalone CLI (periodic captioning).

## Standalone CLI

```bash
uv run -m vision.vlm --backend openai --prompt "What objects are present?" --every-n 30
```

| Flag          | Default                                  | Description                       |
|---------------|------------------------------------------|-----------------------------------|
| `--backend`   | `openai`                                 | VLM backend name                  |
| `--prompt`    | `Describe what is happening...`          | Prompt sent every N frames        |
| `--every-n`   | `30`                                     | Run VLM every N frames            |
| `--max-frames`| `0`                                      | Stop after N frames (`0` = unlimited) |

## Library

```python
from vision.vlm.client import VLMClient

vlm = VLMClient(backend="openai")
resp = vlm.analyze(frame, "Is there a person in the frame?")
print(resp.text, resp.model)
```

## Extending

Add a new backend (e.g. local LLaVA) by subclassing `VisionLanguageModel` and
registering it in `factory.py`. `client.py` and `__main__.py` need no changes.
