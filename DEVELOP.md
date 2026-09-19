# Development Guide

## Overview

A multi-provider LLM SDK for Rust with trait-based architecture supporting Claude, Gemini, Grok, GLM, Groq, OpenRouter, MixLayer, Xiaomi MiMo, Ollama, llama.cpp, OpenAI, and Voyage AI.

**Crate**: `llm-sdk` v0.1.12

**Edition**: 2021

**Toolchain**: nightly (pinned by `rust-toolchain.toml`)

## Architecture

### Core Trait

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError>;
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
    fn supports_streaming(&self) -> bool;
}
```

### Directory Layout

```
src/
├── lib.rs              # Public exports and provider aliases
├── client.rs           # LlmClient trait definition
├── error.rs            # LlmError enum with provider codes
├── types.rs            # CompletionRequest/Response types
├── providers.rs        # Provider name constants
├── models.rs           # Model ID constants (GEMINI_3_PRO, etc.)
├── model_metadata.rs   # Model capabilities and pricing
├── tools/              # Tool/Function calling support
├── claude/             # Anthropic Claude (Messages API)
├── gemini/             # Google Gemini 3 (thinking levels)
├── grok/               # xAI Grok (OpenAI-compatible)
│   ├── xai/            # Paid tier
│   └── zen/            # Free tier (deprecated, use src/zen/)
├── cerebras/           # Cerebras provider API and model exports
├── glm/                # GLM integrations (zAI and Zen)
│   ├── cerebras/       # Paid tier
│   └── zai/            # zAI provider
├── zen/                # OpenCode Zen (multi-lab free models)
├── groq/               # Groq (Chat Completions API)
├── openrouter/         # OpenRouter (multi-model proxy, free model discovery)
├── mixlayer/           # MixLayer (OpenAI-compatible Chat Completions)
├── xiaomi/             # Xiaomi MiMo chat and ASR (OpenAI-compatible Chat Completions)
├── ollama/             # Local models via /api/chat
├── llama_cpp/          # Local models via OpenAI API
├── openai/             # GPT-5 via Responses API
└── voyage/             # Text embeddings

examples/               # 14 runnable examples
tests/                  # 14 integration-test targets and helpers
bin/test_runner.rs      # Automated test runner with TOML config
```

## Key Features

- **Multi-Provider**: Same models via different providers (Zen free tier, xAI/Cerebras paid)
- **Tool Calling**: Type-safe with automatic JSON Schema via schemars
- **Builder Pattern**: Ergonomic request construction
- **Error Handling**: Structured errors (Auth, RateLimit, Network, etc.)
- **Usage Reporting**: Common input/output counts plus optional provider-reported reasoning tokens

## Testing

### Unit Tests
```bash
cargo test --lib
```

### Integration Tests (requires API keys)
```bash
# Manual
ANTHROPIC_API_KEY=xxx cargo test --test claude_integration -- --ignored
OPENROUTER_API_KEY=xxx cargo test --test openrouter_integration -- --ignored
XIAOMI_API_KEY=xxx cargo test --test xiaomi_integration -- --ignored

# ASR test input is raw Base64 for a WAV file (without a data-URL prefix)
XIAOMI_API_KEY=xxx XIAOMI_ASR_AUDIO_BASE64=... \
  cargo test --test xiaomi_integration test_xiaomi_mimo_v2_5_asr -- --ignored

# Or run the targets registered in the test runner with a TOML config
cargo run --bin llm-test-runner --features test-runner -- config.toml
```

The test runner does not currently register the OpenRouter or Xiaomi targets; run those manually
as shown above.

### Config Format (config.toml)
```toml
[api_keys]
anthropic_api_key = "sk-..."
xai_api_key = "xai-..."
openai_api_key = "sk-..."
gemini_api_key = "..."
cerebras_api_key = "..."
zai_api_key = "..."
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| reqwest | HTTP client |
| serde | Serialization |
| tokio | Async runtime |
| thiserror | Error definitions |
| schemars | JSON Schema generation |
| async-trait | Trait async methods |

## Adding a New Provider

1. Create `src/new_provider/` with:
   - `mod.rs` - module exports
   - `client.rs` - struct + `impl LlmClient`
   - `builder.rs` - request builder
   - `types.rs` - request/response types
   - `tools.rs` - tool calling (optional)

2. Add to `src/lib.rs`:
   ```rust
   pub mod new_provider;
   pub use new_provider::NewProviderClient;
   ```

3. Add provider constant to `src/providers.rs`

4. Add model constants to `src/models.rs` and capabilities to `src/model_metadata.rs`

5. Create integration tests in `tests/new_provider_integration.rs`; mark tests that make live API
   calls `#[ignore]` and document the required environment variable.

## Common Tasks

**Run example:**
```bash
cargo run --example simple_completion
```

**Check with all features:**
```bash
cargo check --all-features
```

**Check tests compile:**
```bash
cargo test --no-run
```

**Format and lint:**
```bash
cargo fmt --check
cargo clippy --all-targets --all-features
```

## Model Constants

Located in `src/models.rs`:
- `GEMINI_3_PRO`, `GEMINI_3_FLASH`
- `GROK_CODE_FAST_1`, `GROK_BETA`
- `GPT_5_MINI`, `GPT_5_NANO`
- `VOYAGE_4_LITE`, `VOYAGE_4`
- `ZAI_GLM_4_6`, etc.
- `MIMO_V2_5`, `MIMO_V2_5_PRO`, `MIMO_V2_5_ASR`

OpenRouter has no hardcoded model constants — its free-tier catalog changes frequently, so
`OpenRouterClient::list_free_programming_models()` discovers current free models at runtime
instead.

The provider-neutral `Usage` type in `src/types.rs` exposes `reasoning_tokens: Option<u32>`.
Gemini and OpenAI populate it when their APIs report reasoning-token usage; other providers return
`None` until their response mapping supports an equivalent field.

Xiaomi speech recognition is provider-specific rather than part of `LlmClient::complete()`. The
`XiaomiSpeechRecognitionBuilder` serializes audio content and ASR language options for the shared
Chat Completions endpoint, and maps the transcript through `XiaomiChatCompletionResponse`. Keep
audio wire types and future ASR options in `src/xiaomi/`; `XiaomiAudioFormat` currently supports
WAV and MP3. Live ASR tests must remain ignored and obtain both credentials and audio input from
environment variables rather than committed fixtures.

## Error Codes

- `Authentication` - HTTP 401
- `RateLimit` - HTTP 429 (with retry_after)
- `InvalidRequest` - HTTP 400
- `Api` - Other 4xx/5xx
- `Network` - Connection failures
- `ToolArgumentParse` - Invalid tool params
