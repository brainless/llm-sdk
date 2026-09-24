# Repository Guidelines

## Project Overview

This repository is a Rust 2021 multi-provider LLM SDK. It exposes provider-specific clients and
builders, plus the shared `client::LlmClient` interface and provider-neutral request/response types
in `src/types.rs`. The nightly Rust toolchain is selected by `rust-toolchain.toml`.

## Repository Map

- `src/<provider>/`: provider client, builder, wire types, and tool conversion code.
- `src/client.rs`: shared `LlmClient` trait.
- `src/types.rs`: provider-neutral completion types.
- `src/models.rs`, `src/model_metadata.rs`, `src/providers.rs`: model and provider registry data.
- `src/tools/`: common tool-calling types.
- `tests/`: integration tests; live network tests must be ignored by default.
- `examples/`: runnable usage examples.
- `bin/test_runner.rs`: optional API-key-driven test runner (`test-runner` feature).

## Working Practices

- Preserve unrelated working-tree changes. Inspect `git status` and relevant diffs before editing.
- Keep provider-specific wire behavior inside that provider's module. Map it to common types in the
  provider's `LlmClient` implementation.
- Reuse common `Tool`, `ToolCall`, `ToolChoice`, and `ToolResult` types instead of introducing
  provider-independent duplicates.
- Never commit API keys, tokens, captured response bodies containing sensitive data, or a populated
  test configuration.
- Update `README.md` for user-facing API changes and `DEVELOP.md` for architecture, testing, or
  contributor-workflow changes.

## Adding or Changing a Provider

When applicable, update the provider module, its export in `src/lib.rs`, the canonical name in
`src/providers.rs`, model IDs in `src/models.rs`, and capability entries in
`src/model_metadata.rs`. Add unit tests for serialization and response mapping. Put live API tests
in `tests/<provider>_integration.rs`, guard credentials with environment variables, and mark those
tests `#[ignore]`.

Keep error mapping consistent with `LlmError`: authentication for 401/403, invalid request for
request errors, rate limit for 429, network for transport failures, and API errors for remaining
HTTP failures. Avoid retaining or logging sensitive provider response bodies where sanitized errors
are expected.

## Validation

Run the narrowest relevant test while iterating, then use these checks before handing off:

```bash
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets --no-deps -- -D warnings
```

If a build fails with `sccache: error: Operation not permitted` after a toolchain or dependency
change, remove the stale Cargo probe cache with `rm target/.rustc_info.json` and retry. If the
wrapper still fails, run that check with `RUSTC_WRAPPER=` as a temporary workaround. Do not remove
the whole `target/` directory just for this error.

Use `cargo check --all-features` when changing feature-gated code. Do not run ignored integration
tests unless the required API key is available and live provider calls are explicitly intended.

## Style

Follow `rustfmt`, use descriptive provider-native type names, and prefer typed request/response
structures over ad hoc JSON. Public APIs should have concise rustdoc comments. Tests should assert
serialized wire fields and common-type mappings, especially tools, structured output, token usage,
and provider error status handling.
