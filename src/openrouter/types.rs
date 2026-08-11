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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<OpenRouterProviderPreferences>,
}

/// OpenRouter provider-routing preferences for a chat completion.
///
/// Every field is optional so callers can opt into only the routing guarantees
/// they require while preserving OpenRouter's defaults otherwise.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenRouterProviderPreferences {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<OpenRouterDataCollection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterDataCollection {
    Allow,
    Deny,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    pub choices: Vec<OpenRouterChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenRouterUsage>,
}

/// A parsed completion together with the exact successful response body.
///
/// Keeping the raw bytes lets applications calculate an audit hash without
/// forcing the SDK to choose a hashing algorithm or retain provider payloads.
#[derive(Debug, Clone)]
pub struct OpenRouterChatCompletionResult {
    pub response: OpenRouterChatCompletionResponse,
    pub raw_response: Vec<u8>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_preferences_serialize_with_openrouter_field_names() {
        let request = OpenRouterChatCompletionRequest {
            model: "author/model".into(),
            messages: vec![OpenRouterMessage::user("hello")],
            max_completion_tokens: Some(64),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            provider: Some(OpenRouterProviderPreferences {
                order: Some(vec!["Provider A".into()]),
                allow_fallbacks: Some(false),
                require_parameters: Some(true),
                data_collection: Some(OpenRouterDataCollection::Deny),
                zdr: Some(true),
            }),
        };

        let value = serde_json::to_value(request).unwrap();
        assert_eq!(
            value["provider"],
            serde_json::json!({
                "order": ["Provider A"],
                "allow_fallbacks": false,
                "require_parameters": true,
                "data_collection": "deny",
                "zdr": true
            })
        );
    }

    #[test]
    fn absent_provider_preferences_remain_omitted() {
        let request = OpenRouterChatCompletionRequest {
            model: "author/model".into(),
            messages: vec![],
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            provider: None,
        };

        assert!(serde_json::to_value(request)
            .unwrap()
            .get("provider")
            .is_none());
    }
}
