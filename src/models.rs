//! Model constants for supported LLM providers
//!
//! This module contains official model IDs and human-readable names for all supported providers.
//! Model IDs are sourced from official provider documentation.

/// ElevenLabs speech-to-text model constants.
pub mod elevenlabs {
    /// Scribe v2 batch transcription model.
    pub const SCRIBE_V2_ID: &str = "scribe_v2";
    pub const SCRIBE_V2_NAME: &str = "Scribe v2";
    pub const SCRIBE_V2: &str = SCRIBE_V2_ID;

    /// Scribe v2 Medical batch transcription model.
    pub const SCRIBE_V2_MEDICAL_ID: &str = "scribe_v2_medical";
    pub const SCRIBE_V2_MEDICAL_NAME: &str = "Scribe v2 Medical";
    pub const SCRIBE_V2_MEDICAL: &str = SCRIBE_V2_MEDICAL_ID;
}

/// Claude model constants
pub mod claude {
    /// Claude Sonnet 4.5 - Smart model for complex agents and coding
    /// Released: 2025-09-29
    pub const SONNET_4_5_ID: &str = "claude-sonnet-4-5-20250929";
    pub const SONNET_4_5_NAME: &str = "Claude Sonnet 4.5";

    /// Claude Haiku 4.5 - Fastest model with near-frontier intelligence
    /// Released: 2025-10-01
    pub const HAIKU_4_5_ID: &str = "claude-haiku-4-5-20251001";
    pub const HAIKU_4_5_NAME: &str = "Claude Haiku 4.5";

    /// Claude Opus 4.5 - Premium model combining maximum intelligence with practical performance
    /// Released: 2025-11-01
    pub const OPUS_4_5_ID: &str = "claude-opus-4-5-20251101";
    pub const OPUS_4_5_NAME: &str = "Claude Opus 4.5";

    /// Claude Opus 4.1 - Legacy premium model
    /// Released: 2025-08-05
    pub const OPUS_4_1_ID: &str = "claude-opus-4-1-20250805";
    pub const OPUS_4_1_NAME: &str = "Claude Opus 4.1";

    /// Claude Sonnet 4 - Legacy smart model
    /// Released: 2025-05-14
    pub const SONNET_4_ID: &str = "claude-sonnet-4-20250514";
    pub const SONNET_4_NAME: &str = "Claude Sonnet 4";

    // Backwards compatibility - default to Sonnet 4.5
    pub const SONNET_4_5: &str = SONNET_4_5_ID;
    pub const HAIKU_4_5: &str = HAIKU_4_5_ID;
    pub const OPUS_4_5: &str = OPUS_4_5_ID;
    pub const OPUS_4_1: &str = OPUS_4_1_ID;
    pub const SONNET_4: &str = SONNET_4_ID;
}

/// OpenAI model constants
pub mod openai {
    /// GPT-4o - Latest flagship model
    pub const GPT_4O_ID: &str = "gpt-4o";
    pub const GPT_4O_NAME: &str = "GPT-4o";

    /// GPT-4o Mini - Smaller, faster version of GPT-4o
    pub const GPT_4O_MINI_ID: &str = "gpt-4o-mini";
    pub const GPT_4O_MINI_NAME: &str = "GPT-4o Mini";

    /// GPT-4 Turbo - Enhanced GPT-4 model
    pub const GPT_4_TURBO_ID: &str = "gpt-4-turbo";
    pub const GPT_4_TURBO_NAME: &str = "GPT-4 Turbo";

    /// GPT-4 - Original GPT-4 model
    pub const GPT_4_ID: &str = "gpt-4";
    pub const GPT_4_NAME: &str = "GPT-4";

    /// GPT-3.5 Turbo - Fast and efficient model
    pub const GPT_3_5_TURBO_ID: &str = "gpt-3.5-turbo";
    pub const GPT_3_5_TURBO_NAME: &str = "GPT-3.5 Turbo";

    /// GPT-5 Codex - Advanced coding model (uses Responses API)
    pub const GPT_5_CODEX_ID: &str = "gpt-5-codex";
    pub const GPT_5_CODEX_NAME: &str = "GPT-5 Codex";

    /// GPT-5.1 - Next generation model (uses Responses API)
    pub const GPT_5_1_ID: &str = "gpt-5.1";
    pub const GPT_5_1_NAME: &str = "GPT-5.1";

    /// GPT-5 nano - Compact GPT-5 model
    /// Released: 2025-08-07
    pub const GPT_5_NANO_ID: &str = "gpt-5-nano-2025-08-07";
    pub const GPT_5_NANO_NAME: &str = "GPT-5 nano";

    /// GPT-5 mini - Small GPT-5 model
    /// Released: 2025-08-07
    pub const GPT_5_MINI_ID: &str = "gpt-5-mini-2025-08-07";
    pub const GPT_5_MINI_NAME: &str = "GPT-5 mini";

    // Backwards compatibility
    pub const GPT_4O: &str = GPT_4O_ID;
    pub const GPT_4O_MINI: &str = GPT_4O_MINI_ID;
    pub const GPT_4_TURBO: &str = GPT_4_TURBO_ID;
    pub const GPT_4: &str = GPT_4_ID;
    pub const GPT_3_5_TURBO: &str = GPT_3_5_TURBO_ID;
    pub const GPT_5_CODEX: &str = GPT_5_CODEX_ID;
    pub const GPT_5_1: &str = GPT_5_1_ID;
    pub const GPT_5_NANO: &str = GPT_5_NANO_ID;
    pub const GPT_5_MINI: &str = GPT_5_MINI_ID;
}

/// xAI/Grok model constants
pub mod grok {
    /// Grok Beta - Latest Grok model
    pub const BETA_ID: &str = "grok-beta";
    pub const BETA_NAME: &str = "Grok Beta";

    /// Grok Vision Beta - Grok with vision capabilities
    pub const VISION_BETA_ID: &str = "grok-vision-beta";
    pub const VISION_BETA_NAME: &str = "Grok Vision Beta";

    /// Grok Code Fast 1 - Fast coding model
    pub const CODE_FAST_1_ID: &str = "grok-code-fast-1";
    pub const CODE_FAST_1_NAME: &str = "Grok Code Fast 1";

    // Backwards compatibility
    pub const BETA: &str = BETA_ID;
    pub const VISION_BETA: &str = VISION_BETA_ID;
    pub const CODE_FAST_1: &str = CODE_FAST_1_ID;
}

/// GLM model constants
pub mod glm {
    /// zAI GLM 4.7 - GLM reasoning model via zAI provider (preview)
    pub const ZAI_GLM_4_7_ID: &str = "zai-glm-4.7";
    pub const ZAI_GLM_4_7_NAME: &str = "zAI GLM 4.7";

    /// zAI GLM 4.6 - GLM model via zAI provider
    pub const ZAI_GLM_4_6_ID: &str = "zai-glm-4.6";
    pub const ZAI_GLM_4_6_NAME: &str = "zAI GLM 4.6";

    // Shorthand aliases
    pub const ZAI_GLM_4_7: &str = ZAI_GLM_4_7_ID;
    pub const ZAI_GLM_4_6: &str = ZAI_GLM_4_6_ID;
}

/// Cerebras-hosted model constants
pub mod cerebras {
    /// GPT OSS 120B via Cerebras
    pub const GPT_OSS_120B_ID: &str = "gpt-oss-120b";
    pub const GPT_OSS_120B_NAME: &str = "GPT OSS 120B";

    /// Qwen 3.8 27B via Cerebras
    pub const QWEN_3_8_27B_ID: &str = "qwen-3.8-27b";
    pub const QWEN_3_8_27B_NAME: &str = "Qwen 3.8 27B";

    // Shorthand aliases
    pub const GPT_OSS_120B: &str = GPT_OSS_120B_ID;
    pub const QWEN_3_8_27B: &str = QWEN_3_8_27B_ID;
}

/// Voyage AI embedding model constants
pub mod voyage {
    /// Voyage 4 Large - Highest accuracy embedding model
    /// Default dimension: 1024, supports 256/512/1024/2048
    pub const VOYAGE_4_LARGE_ID: &str = "voyage-4-large";
    pub const VOYAGE_4_LARGE_NAME: &str = "Voyage 4 Large";

    /// Voyage 4 - Balanced performance embedding model
    /// Default dimension: 1024, supports 256/512/1024/2048
    pub const VOYAGE_4_ID: &str = "voyage-4";
    pub const VOYAGE_4_NAME: &str = "Voyage 4";

    /// Voyage 4 Lite - Fast and cost-effective embedding model
    /// Default dimension: 1024, supports 256/512/1024/2048
    /// Max tokens: 1M per batch
    pub const VOYAGE_4_LITE_ID: &str = "voyage-4-lite";
    pub const VOYAGE_4_LITE_NAME: &str = "Voyage 4 Lite";

    /// Voyage 3 Large - Previous generation large model
    /// Default dimension: 1024, supports 256/512/1024/2048
    pub const VOYAGE_3_LARGE_ID: &str = "voyage-3-large";
    pub const VOYAGE_3_LARGE_NAME: &str = "Voyage 3 Large";

    /// Voyage 3.5 - Previous generation balanced model
    /// Default dimension: 1024, supports 256/512/1024/2048
    pub const VOYAGE_3_5_ID: &str = "voyage-3.5";
    pub const VOYAGE_3_5_NAME: &str = "Voyage 3.5";

    /// Voyage 3.5 Lite - Previous generation lite model
    /// Default dimension: 1024, supports 256/512/1024/2048
    pub const VOYAGE_3_5_LITE_ID: &str = "voyage-3.5-lite";
    pub const VOYAGE_3_5_LITE_NAME: &str = "Voyage 3.5 Lite";

    /// Voyage Code 3 - Specialized for code embeddings
    /// Default dimension: 1024, supports 256/512/1024/2048
    pub const VOYAGE_CODE_3_ID: &str = "voyage-code-3";
    pub const VOYAGE_CODE_3_NAME: &str = "Voyage Code 3";

    /// Voyage Finance 2 - Specialized for finance domain
    pub const VOYAGE_FINANCE_2_ID: &str = "voyage-finance-2";
    pub const VOYAGE_FINANCE_2_NAME: &str = "Voyage Finance 2";

    /// Voyage Law 2 - Specialized for legal domain
    pub const VOYAGE_LAW_2_ID: &str = "voyage-law-2";
    pub const VOYAGE_LAW_2_NAME: &str = "Voyage Law 2";

    // Backwards compatibility
    pub const VOYAGE_4_LARGE: &str = VOYAGE_4_LARGE_ID;
    pub const VOYAGE_4: &str = VOYAGE_4_ID;
    pub const VOYAGE_4_LITE: &str = VOYAGE_4_LITE_ID;
    pub const VOYAGE_3_LARGE: &str = VOYAGE_3_LARGE_ID;
    pub const VOYAGE_3_5: &str = VOYAGE_3_5_ID;
    pub const VOYAGE_3_5_LITE: &str = VOYAGE_3_5_LITE_ID;
    pub const VOYAGE_CODE_3: &str = VOYAGE_CODE_3_ID;
    pub const VOYAGE_FINANCE_2: &str = VOYAGE_FINANCE_2_ID;
    pub const VOYAGE_LAW_2: &str = VOYAGE_LAW_2_ID;
}

/// Google Gemini model constants
pub mod gemini {
    /// Gemini 3 Pro - Most intelligent model for complex reasoning
    /// Released: Preview, Context: 1M/64k, Thinking: low/high
    pub const GEMINI_3_PRO_ID: &str = "gemini-3-pro-preview";
    pub const GEMINI_3_PRO_NAME: &str = "Gemini 3 Pro";

    /// Gemini 3 Flash - Pro-level intelligence at Flash speed
    /// Released: Preview, Context: 1M/64k, Thinking: minimal/low/medium/high
    pub const GEMINI_3_FLASH_ID: &str = "gemini-3-flash-preview";
    pub const GEMINI_3_FLASH_NAME: &str = "Gemini 3 Flash";

    // Backwards compatibility
    pub const GEMINI_3_PRO: &str = GEMINI_3_PRO_ID;
    pub const GEMINI_3_FLASH: &str = GEMINI_3_FLASH_ID;
}

/// Ollama local model constants
pub mod ollama {
    /// Ministral 3 3B (Ollama)
    pub const MINISTRAL_3_3B_ID: &str = "ministral-3:3b";
    pub const MINISTRAL_3_3B_NAME: &str = "Ministral 3 3B";

    /// Ministral 3 8B (Ollama)
    pub const MINISTRAL_3_8B_ID: &str = "ministral-3:8b";
    pub const MINISTRAL_3_8B_NAME: &str = "Ministral 3 8B";

    /// Qwen 3.5 2B (Ollama)
    pub const QWEN_3_5_2B_ID: &str = "qwen3.5:2b";
    pub const QWEN_3_5_2B_NAME: &str = "Qwen 3.5 2B";

    /// Qwen 3.5 4B (Ollama)
    pub const QWEN_3_5_4B_ID: &str = "qwen3.5:4b";
    pub const QWEN_3_5_4B_NAME: &str = "Qwen 3.5 4B";

    /// Qwen 3.5 9B (Ollama)
    pub const QWEN_3_5_9B_ID: &str = "qwen3.5:9b";
    pub const QWEN_3_5_9B_NAME: &str = "Qwen 3.5 9B";

    // Backwards compatibility
    pub const MINISTRAL_3_3B: &str = MINISTRAL_3_3B_ID;
    pub const MINISTRAL_3_8B: &str = MINISTRAL_3_8B_ID;
    pub const QWEN_3_5_2B: &str = QWEN_3_5_2B_ID;
    pub const QWEN_3_5_4B: &str = QWEN_3_5_4B_ID;
    pub const QWEN_3_5_9B: &str = QWEN_3_5_9B_ID;
}

/// Groq-hosted model constants
pub mod groq {
    /// OpenAI GPT OSS 120B via Groq — reasoning effort: low/medium/high
    pub const GPT_OSS_120B_ID: &str = "openai/gpt-oss-120b";
    pub const GPT_OSS_120B_NAME: &str = "GPT OSS 120B";

    /// OpenAI GPT OSS 20B via Groq — reasoning effort: low/medium/high
    pub const GPT_OSS_20B_ID: &str = "openai/gpt-oss-20b";
    pub const GPT_OSS_20B_NAME: &str = "GPT OSS 20B";

    // Shorthand aliases
    pub const GPT_OSS_120B: &str = GPT_OSS_120B_ID;
    pub const GPT_OSS_20B: &str = GPT_OSS_20B_ID;
}

/// MixLayer-hosted model constants
pub mod mixlayer {
    /// Qwen 3.5 4B — free tier via MixLayer
    pub const QWEN_3_5_4B_FREE_ID: &str = "qwen/qwen3.5-4b-free";
    pub const QWEN_3_5_4B_FREE_NAME: &str = "Qwen 3.5 4B";

    /// Shorthand alias for the model ID.
    pub const QWEN_3_5_4B_FREE: &str = QWEN_3_5_4B_FREE_ID;
}

/// llama.cpp local model constants (GGUF models)
pub mod llama_cpp {
    /// Qwen 3.5 0.8B GGUF (unsloth/Qwen3.5-0.8B-GGUF:UD-Q4_K_XL)
    /// Small instruction-following model for edge devices
    pub const QWEN_3_5_0_8B_ID: &str = "unsloth/Qwen3.5-0.8B-GGUF:UD-Q4_K_XL";
    pub const QWEN_3_5_0_8B_NAME: &str = "Qwen 3.5 0.8B (GGUF)";

    /// LFM2.5 2.6B GGUF (LiquidAI/LFM2.5-2.6B-GGUF)
    /// Hybrid on-device model by Liquid AI; lfm1.0 license
    pub const LFM_2_5_2_6B_ID: &str = "LiquidAI/LFM2.5-2.6B-GGUF";
    pub const LFM_2_5_2_6B_NAME: &str = "LFM2.5 2.6B (GGUF)";

    // Backwards compatibility
    pub const QWEN_3_5_0_8B: &str = QWEN_3_5_0_8B_ID;
    pub const LFM_2_5_2_6B: &str = LFM_2_5_2_6B_ID;
}

/// Xiaomi MiMo model constants
pub mod xiaomi {
    /// MiMo V2.6 Pro — Xiaomi's flagship reasoning model
    pub const MIMO_V2_6_PRO_ID: &str = "mimo-v2.6-pro";
    pub const MIMO_V2_6_PRO_NAME: &str = "MiMo-V2.6-Pro";

    /// MiMo V2.6 Flash — Xiaomi's efficient reasoning model
    pub const MIMO_V2_6_FLASH_ID: &str = "mimo-v2.6-flash";
    pub const MIMO_V2_6_FLASH_NAME: &str = "MiMo-V2.6-Flash";

    /// MiMo V2.5 — Xiaomi's base reasoning model
    pub const MIMO_V2_5_ID: &str = "mimo-v2.5";
    pub const MIMO_V2_5_NAME: &str = "MiMo V2.5";

    /// MiMo V2.5 Pro — Xiaomi's advanced reasoning model
    pub const MIMO_V2_5_PRO_ID: &str = "mimo-v2.5-pro";
    pub const MIMO_V2_5_PRO_NAME: &str = "MiMo V2.5 Pro";

    /// MiMo V2.5 ASR — Xiaomi's automatic speech recognition model
    pub const MIMO_V2_5_ASR_ID: &str = "mimo-v2.5-asr";
    pub const MIMO_V2_5_ASR_NAME: &str = "MiMo V2.5 ASR";

    // Shorthand aliases
    pub const MIMO_V2_6_PRO: &str = MIMO_V2_6_PRO_ID;
    pub const MIMO_V2_6_FLASH: &str = MIMO_V2_6_FLASH_ID;
    pub const MIMO_V2_5: &str = MIMO_V2_5_ID;
    pub const MIMO_V2_5_PRO: &str = MIMO_V2_5_PRO_ID;
    pub const MIMO_V2_5_ASR: &str = MIMO_V2_5_ASR_ID;
}

/// OpenCode Zen free model constants
pub mod zen {
    /// Big Pickle — stealth model, free on OpenCode for a limited time
    pub const BIG_PICKLE_ID: &str = "big-pickle";
    pub const BIG_PICKLE_NAME: &str = "Big Pickle";

    /// DeepSeek V4 Flash — free on OpenCode Zen
    pub const DEEPSEEK_V4_FLASH_FREE_ID: &str = "deepseek-v4-flash-free";
    pub const DEEPSEEK_V4_FLASH_FREE_NAME: &str = "DeepSeek V4 Flash Free";

    /// MiniMax M2.5 — free on OpenCode Zen
    pub const MINIMAX_M2_5_FREE_ID: &str = "minimax-m2.5-free";
    pub const MINIMAX_M2_5_FREE_NAME: &str = "MiniMax M2.5 Free";

    /// Nemotron 3 Super — free on OpenCode Zen
    pub const NEMOTRON_3_SUPER_FREE_ID: &str = "nemotron-3-super-free";
    pub const NEMOTRON_3_SUPER_FREE_NAME: &str = "Nemotron 3 Super Free";

    // Shorthand aliases
    pub const BIG_PICKLE: &str = BIG_PICKLE_ID;
    pub const DEEPSEEK_V4_FLASH_FREE: &str = DEEPSEEK_V4_FLASH_FREE_ID;
    pub const MINIMAX_M2_5_FREE: &str = MINIMAX_M2_5_FREE_ID;
    pub const NEMOTRON_3_SUPER_FREE: &str = NEMOTRON_3_SUPER_FREE_ID;
}

// Re-export for convenience
pub use claude::*;
