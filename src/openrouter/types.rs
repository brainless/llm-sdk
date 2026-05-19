use serde::{Deserialize, Serialize};

// ─── Chat completion types (OpenAI-compatible) ────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenRouterMessage>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenRouterResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterMessage {
    pub role: OpenRouterRole,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenRouterToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterChatCompletionResponse {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenRouterChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterChoice {
    pub index: u32,
    pub message: OpenRouterMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterErrorResponse {
    pub error: OpenRouterError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterError {
    pub message: String,
    pub code: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterResponseFormatType {
    Text,
    JsonObject,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenRouterResponseFormat {
    #[serde(rename = "type")]
    pub format_type: OpenRouterResponseFormatType,
}

impl OpenRouterResponseFormat {
    pub fn text() -> Self {
        Self {
            format_type: OpenRouterResponseFormatType::Text,
        }
    }

    pub fn json_object() -> Self {
        Self {
            format_type: OpenRouterResponseFormatType::JsonObject,
        }
    }
}

pub type OpenRouterTool = crate::openai::types::OpenAITool;
pub type OpenRouterToolCall = crate::openai::types::OpenAIResponseToolCall;

impl OpenRouterMessage {
    pub fn new<S: Into<String>>(role: OpenRouterRole, content: S) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system<S: Into<String>>(content: S) -> Self {
        Self::new(OpenRouterRole::System, content)
    }

    pub fn user<S: Into<String>>(content: S) -> Self {
        Self::new(OpenRouterRole::User, content)
    }

    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self::new(OpenRouterRole::Assistant, content)
    }

    pub fn tool_result<S: Into<String>>(tool_call_id: S, content: S) -> Self {
        Self {
            role: OpenRouterRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

// ─── Model listing types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModelsResponse {
    pub data: Vec<OpenRouterModelInfo>,
}

/// Subset of model fields relevant to discovery and selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterModelInfo {
    pub id: String,
    pub name: String,
    pub context_length: Option<u64>,
    pub description: Option<String>,
    pub pricing: OpenRouterPricing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterPricing {
    /// Price per million prompt tokens, as a decimal string (e.g. "0" or "0.000001").
    pub prompt: String,
    /// Price per million completion tokens, as a decimal string.
    pub completion: String,
}

impl OpenRouterPricing {
    pub fn is_free(&self) -> bool {
        self.prompt == "0" && self.completion == "0"
    }
}
