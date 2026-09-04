//! Cerebras Chat Completions provider.
//!
//! The implementation historically lived under `glm::cerebras`. Its original
//! names remain exported for compatibility, while new code should use this
//! provider-focused module.

pub use crate::glm::builder::GlmMessageBuilder as CerebrasMessageBuilder;
pub use crate::glm::cerebras::CerebrasGlmClient as CerebrasClient;
pub use crate::glm::tools::GlmToolFormat as CerebrasToolFormat;
pub use crate::glm::types::*;
pub use crate::models::cerebras::*;

pub type CerebrasChatCompletionRequest = crate::glm::types::GlmChatCompletionRequest;
pub type CerebrasChatCompletionResponse = crate::glm::types::GlmChatCompletionResponse;
pub type CerebrasMessage = crate::glm::types::GlmMessage;
pub type CerebrasRole = crate::glm::types::GlmRole;
pub type CerebrasResponseFormat = crate::glm::types::GlmResponseFormat;
