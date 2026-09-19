use serde::{Deserialize, Serialize};

/// A non-streaming Xiaomi Chat Completions request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiChatCompletionRequest {
    pub model: String,
    pub messages: Vec<XiaomiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<XiaomiResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<XiaomiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiMessage {
    pub role: XiaomiRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<XiaomiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum XiaomiRole {
    System,
    User,
    Assistant,
    Tool,
}

impl XiaomiMessage {
    pub fn new(role: XiaomiRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(XiaomiRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(XiaomiRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(XiaomiRole::Assistant, content)
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: XiaomiRole::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<XiaomiChoice>,
    pub usage: XiaomiUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiChoice {
    pub index: u32,
    pub message: XiaomiMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: XiaomiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: XiaomiFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XiaomiFunction {
    pub name: String,
    pub description: String,
    pub parameters: schemars::schema::RootSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum XiaomiResponseFormatType {
    Text,
    JsonObject,
    JsonSchema,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XiaomiResponseFormat {
    #[serde(rename = "type")]
    pub format_type: XiaomiResponseFormatType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<XiaomiJsonSchema>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XiaomiJsonSchema {
    pub schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl XiaomiResponseFormat {
    pub fn text() -> Self {
        Self {
            format_type: XiaomiResponseFormatType::Text,
            json_schema: None,
        }
    }

    pub fn json_object() -> Self {
        Self {
            format_type: XiaomiResponseFormatType::JsonObject,
            json_schema: None,
        }
    }

    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self {
            format_type: XiaomiResponseFormatType::JsonSchema,
            json_schema: Some(XiaomiJsonSchema {
                schema,
                strict: Some(true),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_request() {
        let request = XiaomiChatCompletionRequest {
            model: "mimo-v2.5".into(),
            messages: vec![XiaomiMessage::user("hello")],
            max_completion_tokens: Some(100),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            response_format: None,
            tools: None,
            tool_choice: None,
        };
        let json = serde_json::to_value(request).unwrap();
        assert_eq!(json["model"], "mimo-v2.5");
        assert_eq!(json["max_completion_tokens"], 100);
        assert!(json.get("max_tokens").is_none());
    }

    #[test]
    fn deserializes_response() {
        let raw = r#"{"id":"c1","object":"chat.completion","created":1,"model":"mimo-v2.5","choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}"#;
        let response: XiaomiChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("answer")
        );
        assert_eq!(response.usage.prompt_tokens, 4);
    }
}
