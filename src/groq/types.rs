use serde::{Deserialize, Serialize};

/// Groq chat completion request (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChatCompletionRequest {
    pub model: String,
    pub messages: Vec<GroqMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Reasoning effort for gpt-oss models: "low", "medium" (default), "high"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GroqTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<GroqResponseFormat>,
}

/// A message in the Groq conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMessage {
    pub role: GroqRole,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<GroqToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GroqRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Groq chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChatCompletionResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Vec<GroqChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GroqUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_groq: Option<GroqMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqChoice {
    pub index: u32,
    pub message: GroqMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Groq-specific request metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqMeta {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqErrorResponse {
    pub error: GroqError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroqError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GroqResponseFormatType {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroqResponseFormat {
    #[serde(rename = "type")]
    pub format_type: GroqResponseFormatType,
}

impl GroqResponseFormat {
    pub fn text() -> Self {
        Self { format_type: GroqResponseFormatType::Text }
    }
    pub fn json_object() -> Self {
        Self { format_type: GroqResponseFormatType::JsonObject }
    }
}

pub type GroqTool = crate::openai::types::OpenAITool;
pub type GroqFunction = crate::openai::types::OpenAIFunction;
pub type GroqToolCall = crate::openai::types::OpenAIResponseToolCall;
pub type GroqFunctionCall = crate::openai::types::OpenAIFunctionCall;

impl GroqMessage {
    pub fn new<S: Into<String>>(role: GroqRole, content: S) -> Self {
        Self { role, content: content.into(), tool_calls: None, tool_call_id: None }
    }

    pub fn system<S: Into<String>>(content: S) -> Self {
        Self::new(GroqRole::System, content)
    }

    pub fn user<S: Into<String>>(content: S) -> Self {
        Self::new(GroqRole::User, content)
    }

    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self::new(GroqRole::Assistant, content)
    }

    pub fn assistant_with_tools<S: Into<String>>(content: S, tool_calls: Vec<GroqToolCall>) -> Self {
        Self {
            role: GroqRole::Assistant,
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    pub fn tool_result<S: Into<String>>(tool_call_id: S, content: S) -> Self {
        Self {
            role: GroqRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}
