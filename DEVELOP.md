# Development Guide

## Overview

A multi-provider LLM SDK for Rust with trait-based architecture supporting Claude, Gemini, Grok, GLM, Ollama, llama.cpp, OpenAI, and Voyage AI.

**Crate**: `llm-sdk` v0.1.12  
**Edition**: 2021

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
├── glm/                # Cerebras GLM
│   ├── cerebras/       # Paid tier
│   └── zai/            # zAI provider
├── zen/                # OpenCode Zen (multi-lab free models)
├── ollama/             # Local models via /api/chat
├── llama_cpp/          # Local models via OpenAI API
├── openai/             # GPT-5 via Responses API
└── voyage/             # Text embeddings

examples/               # 16 runnable examples
tests/                  # 13 integration tests
bin/test_runner.rs      # Automated test runner with TOML config
```

## Key Features

- **Multi-Provider**: Same models via different providers (Zen free tier, xAI/Cerebras paid)
- **Tool Calling**: Type-safe with automatic JSON Schema via schemars
- **Builder Pattern**: Ergonomic request construction
- **Error Handling**: Structured errors (Auth, RateLimit, Network, etc.)

## Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests (requires API keys)
```bash
# Manual
ANTHROPIC_API_KEY=xxx cargo test --test claude_integration -- --ignored

# Or use test runner with TOML config
cargo run --bin llm-test-runner --features test-runner -- config.toml
```

### Config Format (config.toml)
```toml
[api_keys]
anthropic_api_key = "sk-..."
xai_api_key = "xai-..."
openai_api_key = "sk-..."
gemini_api_key = "..."
cerebras_api_key = "..."
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

4. Create integration test in `tests/new_provider_integration.rs`

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

## Model Constants

Located in `src/models.rs`:
- `GEMINI_3_PRO`, `GEMINI_3_FLASH`
- `GROK_CODE_FAST_1`, `GROK_BETA`
- `GPT_5_MINI`, `GPT_5_NANO`
- `VOYAGE_4_LITE`, `VOYAGE_4`
- `ZAI_GLM_4_6`, etc.

## Error Codes

- `Authentication` - HTTP 401
- `RateLimit` - HTTP 429 (with retry_after)
- `InvalidRequest` - HTTP 400
- `Api` - Other 4xx/5xx
- `Network` - Connection failures
- `ToolArgumentParse` - Invalid tool params
