use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A non-streaming MixLayer Chat Completions request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerChatCompletionRequest {
    pub model: String,
    pub messages: Vec<MixlayerMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<MixlayerResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<MixlayerTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<MixlayerWebSearchOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerMessage {
    pub role: MixlayerRole,
    /// Text content. Assistant content may be absent for a tool-call response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MixlayerToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sources: Option<Vec<MixlayerSource>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MixlayerRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MixlayerMessage {
    pub fn new(role: MixlayerRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            sources: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MixlayerRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MixlayerRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MixlayerRole::Assistant, content)
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MixlayerRole::Tool,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            sources: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<MixlayerChoice>,
    pub usage: MixlayerUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerChoice {
    pub index: u32,
    pub message: MixlayerMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<MixlayerPromptTokensDetails>,
    #[serde(default)]
    pub server_tool_use: Option<HashMap<String, u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerPromptTokensDetails {
    pub audio_tokens: u32,
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerSource {
    pub title: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: MixlayerFunctionCall,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: MixlayerFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerFunction {
    pub name: String,
    pub description: String,
    pub parameters: schemars::schema::RootSchema,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixlayerResponseFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixlayerResponseFormat {
    #[serde(rename = "type")]
    pub format_type: MixlayerResponseFormatType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<MixlayerJsonSchema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixlayerJsonSchema {
    pub schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl MixlayerResponseFormat {
    pub fn text() -> Self {
        Self {
            format_type: MixlayerResponseFormatType::Text,
            json_schema: None,
        }
    }

    pub fn json_object() -> Self {
        Self {
            format_type: MixlayerResponseFormatType::JsonObject,
            json_schema: None,
        }
    }

    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self {
            format_type: MixlayerResponseFormatType::JsonSchema,
            json_schema: Some(MixlayerJsonSchema {
                schema,
                strict: Some(true),
            }),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MixlayerWebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<MixlayerWebSearchCategory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<MixlayerWebSearchFilters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_characters: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<MixlayerUserLocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MixlayerWebSearchCategory {
    #[serde(rename = "company")]
    Company,
    #[serde(rename = "people")]
    People,
    #[serde(rename = "research paper")]
    ResearchPaper,
    #[serde(rename = "news")]
    News,
    #[serde(rename = "personal site")]
    PersonalSite,
    #[serde(rename = "financial report")]
    FinancialReport,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MixlayerWebSearchFilters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixlayerUserLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    pub approximate: MixlayerApproximateLocation,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MixlayerApproximateLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_mixlayer_specific_options() {
        let request = MixlayerChatCompletionRequest {
            model: "test/model".into(),
            messages: vec![MixlayerMessage::user("hello")],
            max_completion_tokens: Some(100),
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: Some(40),
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: Some(1.1),
            stop: None,
            seed: None,
            reasoning_effort: Some("high".into()),
            thinking: None,
            response_format: Some(MixlayerResponseFormat::json_schema(
                serde_json::json!({"type": "object"}),
            )),
            tools: None,
            tool_choice: None,
            web_search_options: None,
            metadata: None,
            store: Some(false),
        };
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["max_completion_tokens"], 100);
        assert_eq!(json["reasoning_effort"], "high");
        assert_eq!(json["response_format"]["json_schema"]["strict"], true);
        assert!(json.get("max_tokens").is_none());
    }

    #[test]
    fn serializes_thinking_false_to_disable_thinking() {
        let request = MixlayerChatCompletionRequest {
            model: "test/model".into(),
            messages: vec![MixlayerMessage::user("hello")],
            max_completion_tokens: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop: None,
            seed: None,
            reasoning_effort: None,
            thinking: Some(false),
            response_format: None,
            tools: None,
            tool_choice: None,
            web_search_options: None,
            metadata: None,
            store: None,
        };
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["thinking"], false);
    }

    #[test]
    fn deserializes_reasoning_sources_and_usage_details() {
        let raw = r#"{"id":"c1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"answer","reasoning_content":"thought","sources":[{"title":"Docs","url":"https://example.com"}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":1},"server_tool_use":{"web_search":1}}}"#;
        let response: MixlayerChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let message = &response.choices[0].message;
        assert_eq!(message.reasoning_content.as_deref(), Some("thought"));
        assert_eq!(message.sources.as_ref().unwrap()[0].title, "Docs");
        assert_eq!(
            response
                .usage
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .cached_tokens,
            1
        );
    }
}
