# AI Voice Ordering PoC — Portfolio Repository Design

## Purpose

Turn the public `AI_Ordering_PoC` repository into a credible portfolio artifact for AI Engineer and Forward Deployed Engineer roles. A reviewer should be able to understand the problem, architecture, engineering decisions, and the author's leadership contribution within five minutes, then verify the core behavior locally without AWS or Twilio credentials.

The repository represents a sanitized proof of concept. It may state that the author served as AI Engineer Lead for a six-person team and led the solution from 0→1 PoC to production. It must not include production code, architecture, customer information, business data, or performance metrics.

## Success Criteria

1. The README explains the system and the author's role in under one screen before presenting setup details.
2. The application is organized into focused modules with explicit interfaces around external services.
3. Configuration contains no fixed IP addresses, machine-specific paths, credentials, or private service identifiers.
4. Concurrent calls do not share conversation state.
5. Invalid model output cannot be sent to the order API.
6. Order submission uses a stable idempotency key and bounded network calls.
7. Core orchestration, parsing, session isolation, webhook behavior, and error paths run in automated tests without network access.
8. A GitHub Actions workflow runs formatting/lint and tests on every push and pull request.
9. Repository description and topics accurately represent the sanitized PoC.

## Public Narrative

The README will use this order:

1. **Project title and one-line value proposition** — a real-time phone ordering assistant that transcribes speech, reasons over a constrained menu, returns spoken responses, and submits validated structured orders.
2. **Leadership and disclosure** — the author was AI Engineer Lead for a six-person team and led the solution from 0→1 PoC to production; this repository contains only the sanitized PoC.
3. **System flow** — a Mermaid diagram and a concise numbered walkthrough.
4. **Engineering highlights** — streaming audio handling, per-call state, structured LLM output, menu validation, adapter boundaries, and safe downstream integration.
5. **Quick start** — Python environment, system dependencies, Vosk model, environment configuration, test command, and server command.
6. **Configuration** — documented environment variables with safe examples.
7. **Testing and design decisions** — what can be verified offline and why the boundaries exist.
8. **Project structure and limitations** — honest PoC constraints and high-value next steps.

The README will not claim unverified latency, accuracy, traffic, conversion, savings, or production architecture. Sunrise, Uptrillion, fixed infrastructure addresses, and host-specific paths will be replaced with generic examples.

## Architecture

```text
Twilio webhook / bidirectional Media Stream
                    │
                    ▼
             ConversationService
            ┌───────┼────────┐
            ▼       ▼        ▼
           ASR     LLM      TTS
            │       │        │
            └───────┼────────┘
                    ▼
               OrderGateway
```

### Components

- `app.py` owns Flask routes, Twilio TwiML, WebSocket event decoding, and translation between transport events and application calls.
- `conversation.py` owns one call's transcript, turn handling, silence-triggered processing, response generation, and order submission flow.
- `models.py` defines menu items, order lines, validated orders, and assistant turns.
- `config.py` reads environment variables once and validates required settings at startup.
- `adapters/bedrock.py` builds the constrained prompt, calls Amazon Bedrock, and converts model output into an assistant turn.
- `adapters/speech.py` wraps Vosk transcription, text-to-speech, and telephony audio conversion.
- `adapters/order_gateway.py` submits orders with timeout, idempotency, and explicit error mapping.
- `ai_ordering_poc.py` remains a small compatibility entry point so the repository retains an obvious launcher.

Adapters expose small protocols so the core service can be tested with deterministic fakes. External SDK clients are created lazily rather than at module import time, allowing tests and documentation tools to import the package without credentials.

## Data Flow

1. Twilio posts to `/call`; the application returns TwiML that connects a bidirectional Media Stream to `/media`.
2. A `start` event creates a session keyed by `streamSid` and returns a welcome message.
3. `media` events are decoded from base64 µ-law, normalized for Vosk, and passed to the session's recognizer.
4. A final transcript plus a configurable silence interval triggers one conversational turn.
5. The LLM receives the menu, conversation context, and a strict response contract.
6. The returned payload is parsed and validated. Item names must exist on the configured menu, quantities must be positive integers, and server-side prices replace any model-supplied prices.
7. An incomplete conversation returns only a short spoken response. A complete valid order is sent through `OrderGateway` using an idempotency key derived from the call and validated order.
8. The final response is synthesized, encoded as telephony audio, and returned over the WebSocket.
9. A `stop` event or socket close removes the session and temporary audio resources.

## Configuration and Security

`.env.example` will document safe placeholders for:

- AWS region and Bedrock model ID
- Vosk model directory
- order API base URL and optional bearer token
- public WebSocket base URL
- silence threshold, request timeout, and log level
- optional Twilio/ngrok development settings

Secrets remain in environment variables and are never logged. Logs include event type, stream ID, duration, and outcome but exclude raw audio, credentials, and full order payloads by default. The downstream URL must use HTTPS outside local development.

No open-source license will be added in this change. Public visibility does not establish redistribution rights, and the production implementation is intentionally excluded. Licensing can be handled separately after ownership is confirmed.

## Error Handling

- Malformed WebSocket events are logged and ignored without terminating unrelated sessions.
- Missing configuration fails fast with a concise startup error.
- ASR, LLM, TTS, and order failures map to distinct application exceptions and user-safe replies.
- Network operations use configurable timeouts; automatic retries are limited to operations that are safe to repeat.
- Order requests carry an idempotency key to prevent duplicate submission.
- Invalid or hallucinated menu items never reach the downstream order API.
- Session and temporary-file cleanup runs on normal stop, socket closure, and exceptions.

## Verification Strategy

The package will use `pytest` and deterministic fake adapters. Tests will cover:

- valid and invalid structured LLM responses
- menu allow-list and server-authoritative pricing
- incomplete versus complete conversations
- two simultaneous calls maintaining separate history
- Twilio `start`, `media`, and `stop` event behavior
- order API success, timeout, rejection, and idempotency headers
- configuration validation
- application import and health endpoint without cloud credentials

`ruff` will check formatting and common correctness issues. GitHub Actions will run lint and tests on supported Python versions. Live AWS/Twilio tests remain optional because they require credentials, infrastructure, a Vosk model, and telephony access.

## Repository Layout

```text
.
├── .env.example
├── .github/workflows/ci.yml
├── .gitignore
├── README.md
├── ai_ordering_poc.py
├── pyproject.toml
├── src/ai_ordering/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── conversation.py
│   ├── models.py
│   └── adapters/
│       ├── bedrock.py
│       ├── order_gateway.py
│       └── speech.py
└── tests/
    ├── test_app.py
    ├── test_config.py
    ├── test_conversation.py
    ├── test_models.py
    └── test_order_gateway.py
```

The repository will not include model binaries, recorded calls, generated audio, credentials, customer data, deployment manifests for production, or reconstructed production internals.

## GitHub Presentation

After local verification, update the repository description to describe the real-time AI voice ordering PoC and add focused topics such as `ai`, `voice-ai`, `aws-bedrock`, `twilio`, `vosk`, `flask`, and `proof-of-concept`. The repository name and visibility will remain unchanged.

## Non-Goals

- Reproducing or disclosing the production system
- Inventing performance or business metrics
- Adding a browser UI unrelated to the phone-ordering flow
- Bundling large speech models or audio samples
- Adding infrastructure-as-code that cannot be verified from the public PoC
- Claiming production readiness for this public repository

