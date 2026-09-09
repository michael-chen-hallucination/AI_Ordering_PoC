# AI Voice Ordering PoC Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the public single-file proof of concept into a sanitized, testable, recruiter-ready AI voice ordering repository without disclosing production implementation details.

**Architecture:** A Flask/Twilio transport layer delegates each call to an isolated `ConversationService`. Small adapters wrap Amazon Bedrock, Vosk/gTTS, and the downstream order API; typed domain models validate all model output before side effects occur.

**Tech Stack:** Python 3.11–3.12, Flask, Flask-Sock, Twilio, Amazon Bedrock via boto3, Vosk, gTTS, pydub, requests, pytest, Ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-08-portfolio-repository-polish-design.md`

## Global Constraints

- The repository is a sanitized PoC; do not include production code, architecture, customer information, business data, or performance metrics.
- State that the author was AI Engineer Lead for a six-person team and led the solution from 0→1 PoC to production.
- Remove Sunrise, Uptrillion, fixed IP addresses, host-specific paths, credentials, and private service identifiers.
- Support Python `>=3.11,<3.13` because the telephony codec path uses the standard-library `audioop` module.
- Do not require AWS, Twilio, a Vosk model, or network access for the default automated test suite.
- Do not add an open-source license in this change.
- Do not add model binaries, recorded calls, generated audio, production deployment manifests, or reconstructed production internals.

---

### Task 1: Package Foundation, Configuration, and Validated Domain Models

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/ai_ordering/__init__.py`
- Create: `src/ai_ordering/config.py`
- Create: `src/ai_ordering/models.py`
- Create: `tests/test_config.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Produces: `Settings.from_env(environ: Mapping[str, str] | None = None) -> Settings`
- Produces: `Settings.websocket_url(request_host: str) -> str`
- Produces: `Menu.from_mapping(values: Mapping[str, str]) -> Menu`
- Produces: `parse_assistant_payload(raw: str, menu: Menu) -> AssistantTurn`
- Produces: immutable `ChatMessage`, `MenuItem`, `OrderLine`, `Order`, and `AssistantTurn` dataclasses.

- [ ] **Step 1: Add packaging and development dependencies**

Create `pyproject.toml` with a `src` package layout, runtime dependencies (`boto3`, `Flask`, `flask-sock`, `gTTS`, `pydub`, `requests`, `twilio`, `vosk`), and a `dev` extra containing `pytest`, `pytest-cov`, and `ruff`. Configure pytest with `pythonpath = ["src"]` and Ruff for Python 3.11 with an 88-character line length.

- [ ] **Step 2: Write failing configuration and model tests**

Cover safe defaults, environment overrides, WebSocket URL derivation, valid order parsing, fenced JSON parsing, unknown menu items, invalid quantities, and server-authoritative prices. Representative assertions:

```python
def test_model_price_is_replaced_by_menu_price(menu):
    turn = parse_assistant_payload(
        '{"message":"Confirmed","order":{"complete":true,'
        '"items":[{"name":"Coke","quantity":2,"price":"0.01"}]}}',
        menu,
    )
    assert turn.order.total == Decimal("4.00")


def test_unknown_menu_item_is_rejected(menu):
    with pytest.raises(ModelOutputError, match="not on the menu"):
        parse_assistant_payload(
            '{"message":"Okay","order":{"complete":true,'
            '"items":[{"name":"Pizza","quantity":1}]}}',
            menu,
        )
```

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_config.py tests/test_models.py -q`

Expected: collection fails because `ai_ordering.config` and `ai_ordering.models` do not exist.

- [ ] **Step 4: Implement settings and domain validation**

Use frozen dataclasses and `Decimal` for money. `parse_assistant_payload` must strip an optional Markdown JSON fence, require a non-empty `message`, accept no order for an incomplete turn, require at least one line for a complete order, resolve item names case-insensitively against `Menu`, reject boolean/non-integer/non-positive quantities, ignore model prices, and merge duplicate menu lines.

The settings object must expose these fields with safe defaults:

```python
@dataclass(frozen=True)
class Settings:
    aws_region: str = "us-east-1"
    bedrock_model_id: str = ""
    vosk_model_path: str = "models/vosk-model-small-en-us-0.15"
    order_api_url: str = ""
    order_api_token: str = ""
    public_websocket_url: str = ""
    silence_seconds: float = 1.0
    request_timeout_seconds: float = 10.0
    log_level: str = "INFO"
    menu: Menu = DEFAULT_MENU
```

`websocket_url()` uses the explicit public URL when supplied; otherwise it converts an HTTPS request host to `wss://.../media` and HTTP to `ws://.../media`.

- [ ] **Step 5: Add safe configuration examples and ignores**

`.env.example` contains only documented placeholders. `.gitignore` excludes `.env`, virtual environments, Python caches, coverage output, Ruff/pytest caches, model directories, audio files, and local logs.

- [ ] **Step 6: Run tests and lint**

Run: `python -m pytest tests/test_config.py tests/test_models.py -q`

Expected: all focused tests pass.

Run: `python -m ruff check src tests`

Expected: no violations.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/ai_ordering tests/test_config.py tests/test_models.py
git commit -m "feat: add validated ordering domain"
```

### Task 2: Per-Call Conversation Orchestration

**Files:**
- Create: `src/ai_ordering/conversation.py`
- Create: `tests/test_conversation.py`

**Interfaces:**
- Consumes: `ChatMessage`, `AssistantTurn`, and `Order` from `ai_ordering.models`.
- Produces: `Assistant.respond(history: Sequence[ChatMessage]) -> AssistantTurn` protocol.
- Produces: `OrderSink.submit(order: Order, idempotency_key: str) -> None` protocol.
- Produces: `ConversationService.start(stream_id: str) -> AssistantTurn`
- Produces: `ConversationService.handle_transcript(stream_id: str, transcript: str) -> AssistantTurn`
- Produces: `ConversationService.stop(stream_id: str) -> None`
- Produces: `ConversationService.active_session_count -> int`

- [ ] **Step 1: Write failing orchestration tests**

Use fake assistant and order sink classes. Cover welcome behavior, empty transcript rejection, independent histories for two stream IDs, order submission only for complete orders, stable idempotency keys, no duplicate submission for the same completed order, and session cleanup.

```python
def test_two_calls_do_not_share_history(service, assistant):
    service.start("call-a")
    service.start("call-b")
    service.handle_transcript("call-a", "two cokes")
    service.handle_transcript("call-b", "one burger")
    assert assistant.customer_texts_by_call == [
        ["two cokes"],
        ["one burger"],
    ]


def test_completed_order_is_submitted_once(service, order_sink):
    service.start("call-a")
    service.handle_transcript("call-a", "two cokes")
    service.handle_transcript("call-a", "two cokes")
    assert len(order_sink.calls) == 1
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_conversation.py -q`

Expected: collection fails because `ai_ordering.conversation` does not exist.

- [ ] **Step 3: Implement the service with isolated state**

Store `CallSession(history: list[ChatMessage], submitted_keys: set[str])` in a private dictionary keyed by stream ID. Add each customer message before calling the assistant, append the assistant response afterward, and derive the idempotency key as the first 32 hexadecimal characters of SHA-256 over `stream_id` plus `order.canonical_json()`.

Raise `UnknownSessionError` for transcripts received before `start()`. Return a welcome `AssistantTurn` from `start()` without calling external adapters. Treat repeated `start()` for the same stream ID as a state reset.

- [ ] **Step 4: Run tests and lint**

Run: `python -m pytest tests/test_conversation.py -q`

Expected: all orchestration tests pass.

Run: `python -m ruff check src/ai_ordering/conversation.py tests/test_conversation.py`

Expected: no violations.

- [ ] **Step 5: Commit**

```bash
git add src/ai_ordering/conversation.py tests/test_conversation.py
git commit -m "feat: isolate call conversation state"
```

### Task 3: Safe Downstream Order Adapter

**Files:**
- Create: `src/ai_ordering/adapters/__init__.py`
- Create: `src/ai_ordering/adapters/order_gateway.py`
- Create: `tests/test_order_gateway.py`

**Interfaces:**
- Consumes: `Order` from `ai_ordering.models`.
- Produces: `HttpOrderGateway.submit(order: Order, idempotency_key: str) -> None`
- Produces: `OrderGatewayError` for timeouts, request errors, and non-success responses.

- [ ] **Step 1: Write failing HTTP adapter tests**

Inject a fake `requests.Session`. Assert the JSON body contains item name, quantity, unit price, and total; assert the `Idempotency-Key` header, optional bearer token, and configured timeout; cover 201 success, timeout, and 400/500 failures.

```python
def test_submit_sets_timeout_and_idempotency(order, fake_session):
    gateway = HttpOrderGateway(
        "https://orders.example.com/v1/orders",
        token="test-token",
        timeout_seconds=3.5,
        session=fake_session,
    )
    gateway.submit(order, "call-key")
    assert fake_session.last_headers["Idempotency-Key"] == "call-key"
    assert fake_session.last_timeout == 3.5
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_order_gateway.py -q`

Expected: collection fails because the adapter does not exist.

- [ ] **Step 3: Implement the HTTP adapter**

Reject a missing URL during adapter construction. Send one POST per `submit()` call. Map `requests.Timeout`, other `RequestException` values, and non-2xx status codes to `OrderGatewayError` without including credentials or response bodies in exception messages.

- [ ] **Step 4: Run tests and lint**

Run: `python -m pytest tests/test_order_gateway.py -q`

Expected: all adapter tests pass.

Run: `python -m ruff check src/ai_ordering/adapters/order_gateway.py tests/test_order_gateway.py`

Expected: no violations.

- [ ] **Step 5: Commit**

```bash
git add src/ai_ordering/adapters tests/test_order_gateway.py
git commit -m "feat: add idempotent order gateway"
```

### Task 4: Bedrock Assistant Adapter

**Files:**
- Create: `src/ai_ordering/adapters/bedrock.py`
- Create: `tests/test_bedrock.py`

**Interfaces:**
- Consumes: `ChatMessage`, `Menu`, `AssistantTurn`, and `parse_assistant_payload`.
- Produces: `BedrockAssistant.respond(history: Sequence[ChatMessage]) -> AssistantTurn`
- Produces: `BedrockError` for SDK errors, empty responses, and invalid response shape.

- [ ] **Step 1: Write failing Bedrock adapter tests**

Inject a fake Bedrock Runtime client. Verify that `converse()` receives the configured model ID, a system prompt containing the menu and strict JSON contract, translated user/assistant history, and deterministic inference settings. Cover valid output, fenced JSON, missing model ID, SDK failure, and malformed content blocks.

```python
def test_respond_uses_converse_and_validates_output(menu, fake_client):
    fake_client.response = {
        "output": {"message": {"content": [{"text": VALID_RESPONSE}]}}
    }
    assistant = BedrockAssistant("model-id", menu, client=fake_client)
    turn = assistant.respond([ChatMessage(role="user", content="two cokes")])
    assert turn.order.items[0].quantity == 2
    assert fake_client.last_request["modelId"] == "model-id"
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_bedrock.py -q`

Expected: collection fails because the adapter does not exist.

- [ ] **Step 3: Implement the Bedrock Converse adapter**

Create the boto3 client lazily when `respond()` is first called if none was injected. Use `client.converse()` with the configured model ID, `temperature: 0.1`, `topP: 0.9`, and `maxTokens: 400`. The system prompt must require exactly this semantic shape:

```json
{
  "message": "short spoken reply",
  "order": {
    "complete": true,
    "items": [{"name": "exact menu name", "quantity": 1}]
  }
}
```

For an incomplete order, `order` is `null`. Pass the returned text through `parse_assistant_payload()` so hallucinated items and quantities fail before the order gateway is called.

- [ ] **Step 4: Run tests and lint**

Run: `python -m pytest tests/test_bedrock.py -q`

Expected: all Bedrock adapter tests pass.

Run: `python -m ruff check src/ai_ordering/adapters/bedrock.py tests/test_bedrock.py`

Expected: no violations.

- [ ] **Step 5: Commit**

```bash
git add src/ai_ordering/adapters/bedrock.py tests/test_bedrock.py
git commit -m "feat: add structured Bedrock assistant"
```

### Task 5: Streaming Speech Adapters

**Files:**
- Create: `src/ai_ordering/adapters/speech.py`
- Create: `tests/test_speech.py`

**Interfaces:**
- Produces: `StreamingTranscriber.push_ulaw(payload: bytes, now: float | None = None) -> str | None`
- Produces: `StreamingTranscriber.close() -> None`
- Produces: `VoskTranscriberFactory.create() -> StreamingTranscriber`
- Produces: `SpeechSynthesizer.synthesize_ulaw(text: str) -> str` returning base64-encoded 8 kHz µ-law audio.
- Produces: `SpeechError` for model loading, recognition, synthesis, and audio conversion failures.

- [ ] **Step 1: Write failing speech adapter tests**

Inject a fake Vosk recognizer and fake clock. Verify µ-law decoding, resampling from 8 kHz to the configured recognizer rate, final transcript extraction after the silence threshold, blank transcript suppression, and recognizer finalization. Patch gTTS and `AudioSegment.from_file` to verify that the synthesizer returns base64 µ-law bytes without writing temporary files.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_speech.py -q`

Expected: collection fails because the speech adapter does not exist.

- [ ] **Step 3: Implement streaming transcription**

Decode Twilio audio with `audioop.ulaw2lin(payload, 2)` and `audioop.ratecv(..., 8000, sample_rate, None)`. Keep pending final text and the last observed speech time. Return the pending text only after `silence_seconds` has elapsed; return `None` for partial or empty results. Construct Vosk models in `VoskTranscriberFactory.create()` so application import does not load a large model.

- [ ] **Step 4: Implement in-memory text-to-speech conversion**

Write gTTS MP3 output to `io.BytesIO`, decode it with pydub, normalize to mono 8 kHz 16-bit PCM, convert with `audioop.lin2ulaw`, and return ASCII base64. Never create `temp.mp3` or use a machine-specific path.

- [ ] **Step 5: Run tests and lint**

Run: `python -m pytest tests/test_speech.py -q`

Expected: all speech adapter tests pass.

Run: `python -m ruff check src/ai_ordering/adapters/speech.py tests/test_speech.py`

Expected: no violations.

- [ ] **Step 6: Commit**

```bash
git add src/ai_ordering/adapters/speech.py tests/test_speech.py
git commit -m "feat: add streaming speech adapters"
```

### Task 6: Flask, Twilio, and WebSocket Transport

**Files:**
- Create: `src/ai_ordering/app.py`
- Create: `tests/test_app.py`
- Replace: `ai_ordering_poc.py`

**Interfaces:**
- Consumes: `Settings`, `ConversationService`, `VoskTranscriberFactory`, and `SpeechSynthesizer`.
- Produces: `MediaStreamHandler.handle(message: str) -> list[str]`
- Produces: `MediaStreamHandler.disconnect(stream_id: str) -> None`
- Produces: `create_app(settings: Settings | None = None, handler: MediaStreamHandler | None = None) -> Flask`
- Produces: `build_default_handler(settings: Settings) -> MediaStreamHandler`

- [ ] **Step 1: Write failing transport tests**

Cover `GET /healthz`, Twilio-compatible XML from `POST /call`, `start` producing a welcome media message, `media` forwarding audio and returning synthesized assistant audio, malformed events being ignored, `stop` cleanup, and two stream IDs using separate transcribers.

```python
def test_call_webhook_returns_bidirectional_stream(app):
    response = app.test_client().post("/call")
    assert response.status_code == 200
    assert b"<Connect>" in response.data
    assert b'wss://voice.example.com/media' in response.data


def test_health_does_not_require_cloud_credentials(app):
    response = app.test_client().get("/healthz")
    assert response.get_json() == {"status": "ok"}
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run: `python -m pytest tests/test_app.py -q`

Expected: collection fails because `ai_ordering.app` does not exist.

- [ ] **Step 3: Implement the media handler**

Parse each message defensively. On `start`, validate `streamSid`, create a transcriber, call `ConversationService.start()`, and emit one Twilio `media` JSON message. On `media`, base64-decode the payload, pass it to the stream's transcriber, and when a final transcript is returned call `handle_transcript()` and emit synthesized audio. On `stop`, close and remove the transcriber and conversation session.

Log event type and stream ID without logging audio or full orders. Convert known speech, model, and gateway errors into a generic spoken retry message; malformed events return no outbound messages.

- [ ] **Step 4: Implement Flask and Twilio routes**

`POST /call` returns `VoiceResponse` with `Connect().stream(url=settings.websocket_url(request.host_url))`. `GET /healthz` returns `{"status": "ok"}`. `/media` receives Flask-Sock messages until stop or disconnect, sends every outbound message, and always performs cleanup in `finally`.

`build_default_handler()` wires lazy Bedrock, Vosk, gTTS, and HTTP adapters. It validates the Bedrock model ID and order API URL only when the real handler is built, not when the package or health route is imported.

- [ ] **Step 5: Replace the compatibility launcher**

Make `ai_ordering_poc.py` a small entry point:

```python
from ai_ordering.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8443)
```

Do not manage ngrok or mutate a Twilio phone number at startup; README setup will explain how to configure the webhook explicitly.

- [ ] **Step 6: Run tests and lint**

Run: `python -m pytest tests/test_app.py -q`

Expected: all transport tests pass.

Run: `python -m ruff check src/ai_ordering/app.py tests/test_app.py ai_ordering_poc.py`

Expected: no violations.

- [ ] **Step 7: Commit**

```bash
git add src/ai_ordering/app.py tests/test_app.py ai_ordering_poc.py
git commit -m "feat: add Twilio media stream transport"
```

### Task 7: Recruiter-Facing Documentation, CI, and Final Verification

**Files:**
- Modify: `README.md`
- Create: `.github/workflows/ci.yml`
- Modify: `docs/superpowers/plans/2026-09-08-portfolio-repository-polish.md`

**Interfaces:**
- Consumes: all commands, modules, configuration keys, and behavior implemented in Tasks 1–6.
- Produces: a five-minute repository walkthrough, reproducible setup instructions, and automated CI verification.

- [ ] **Step 1: Write the README**

Include the approved public narrative, a Mermaid system diagram, end-to-end flow, engineering highlights, repository structure, prerequisites, exact setup and test commands, environment variable table, Twilio webhook setup, privacy disclosure, limitations, and next steps. Use this leadership statement without expanding it into unverified claims:

> I served as AI Engineer Lead for a six-person team, leading the solution from 0→1 PoC through production delivery. This public repository contains only a sanitized PoC; production code, architecture, customer data, and business metrics are intentionally excluded.

- [ ] **Step 2: Add GitHub Actions CI**

Run on pushes and pull requests using Python 3.11 and 3.12. Install `.[dev]`, run `ruff check .`, run `ruff format --check .`, and run `pytest -q`. Do not configure live integration credentials.

- [ ] **Step 3: Run the complete local verification suite**

Run: `python -m ruff check .`

Expected: no violations.

Run: `python -m ruff format --check .`

Expected: all files formatted.

Run: `python -m pytest -q`

Expected: all tests pass without network access or cloud credentials.

Run: `python -m compileall -q src ai_ordering_poc.py`

Expected: exit status 0.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 4: Verify public-content boundaries**

Run:

```bash
rg -n "Sunrise|Uptrillion|54\.174\.193\.122|/home/ec2-user|AKIA|secret|password" \
  --glob '!docs/superpowers/**' .
```

Expected: no private identifiers, fixed infrastructure addresses, host paths, or credentials. Generic `.env.example` variable names and explanatory security prose are acceptable when manually reviewed.

- [ ] **Step 5: Commit documentation and CI**

```bash
git add README.md .github/workflows/ci.yml docs/superpowers/plans/2026-09-08-portfolio-repository-polish.md
git commit -m "docs: present AI voice ordering PoC"
```

- [ ] **Step 6: Update GitHub repository metadata**

Run:

```bash
gh repo edit michael-chen-hallucination/AI_Ordering_PoC \
  --description "Real-time AI voice ordering PoC using Twilio, Vosk, and Amazon Bedrock" \
  --add-topic ai \
  --add-topic voice-ai \
  --add-topic aws-bedrock \
  --add-topic twilio \
  --add-topic vosk \
  --add-topic flask \
  --add-topic proof-of-concept
```

Expected: description and topics are visible in `gh repo view` output; name and visibility are unchanged.

- [ ] **Step 7: Push and verify the remote branch**

Run: `git push origin main`

Expected: push succeeds.

Run: `git status --short --branch`

Expected: `main` is clean and aligned with `origin/main`.

Run: `gh repo view michael-chen-hallucination/AI_Ordering_PoC --json description,repositoryTopics,url`

Expected: the recruiter-facing description, seven focused topics, and correct public URL are returned.

