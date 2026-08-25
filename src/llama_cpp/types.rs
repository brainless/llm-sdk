use serde::{Deserialize, Serialize};

use crate::openai::types::OpenAITool;

/// Llama.cpp chat completion request (OpenAI-compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppChatCompletionRequest {
    pub model: String,
    pub messages: Vec<LlamaCppMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// Llama.cpp chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<LlamaCppChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<LlamaCppUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppChoice {
    pub index: u32,
    pub message: LlamaCppMessage,
    #[serde(rename = "finish_reason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppUsage {
    #[serde(rename = "prompt_tokens")]
    pub prompt_tokens: u32,
    #[serde(rename = "completion_tokens")]
    pub completion_tokens: u32,
    #[serde(rename = "total_tokens")]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppMessage {
    pub role: LlamaCppRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Reasoning/thinking content emitted alongside `content` by reasoning
    /// models served through llama.cpp's OpenAI-compatible endpoint (a
    /// documented llama.cpp server extension field). A model can spend its
    /// entire token budget here and return no final `content` at all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<LlamaCppToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl LlamaCppMessage {
    pub fn new(role: LlamaCppRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: LlamaCppRole::Tool,
            content: Some(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlamaCppRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LlamaCppToolCall {
    /// Simplified tool call format (name + arguments string)
    Simple { name: String, arguments: String },
    /// OpenAI-style tool call format
    OpenAI(crate::openai::types::OpenAIResponseToolCall),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_reasoning_content_alongside_empty_final_content() {
        let raw = r#"{
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "small-reasoner",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "reasoning_content": "thinking forever without ever answering..."
                },
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 512,
                "total_tokens": 522
            }
        }"#;

        let response: LlamaCppChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = response.choices.first().unwrap();

        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
        assert_eq!(choice.message.content, None);
        assert_eq!(
            choice.message.reasoning_content.as_deref(),
            Some("thinking forever without ever answering...")
        );
        assert_eq!(response.usage.as_ref().unwrap().completion_tokens, 512);
    }

    #[test]
    fn deserializes_without_reasoning_content_for_backward_compat() {
        let raw = r#"{
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "created": 1,
            "model": "plain-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello"
                },
                "finish_reason": "stop"
            }]
        }"#;

        let response: LlamaCppChatCompletionResponse = serde_json::from_str(raw).unwrap();
        let choice = response.choices.first().unwrap();

        assert_eq!(choice.message.content.as_deref(), Some("hello"));
        assert_eq!(choice.message.reasoning_content, None);
        assert!(response.usage.is_none());
    }
}
