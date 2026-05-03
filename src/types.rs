use serde::{Deserialize, Serialize};

/// Role of a message in a conversation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// User message
    User,
    /// Assistant message
    Assistant,
    /// System message
    System,
    /// Tool message (result for a previous tool call)
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::System => write!(f, "system"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// Content block for multimodal messages
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentBlock {
    /// Text content
    Text { text: String },
    /// Image content (for future multimodal support)
    Image {
        #[serde(rename = "type")]
        content_type: String,
        source: ImageSource,
    },
}

/// Image source for multimodal content
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageSource {
    /// Type of image source (e.g., "base64")
    #[serde(rename = "type")]
    pub source_type: String,
    /// Media type (e.g., "image/jpeg")
    #[serde(rename = "media_type")]
    pub media_type: String,
    /// Base64 encoded image data
    pub data: String,
}

/// A message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Role of the message sender
    pub role: Role,
    /// Content of the message
    pub content: Vec<ContentBlock>,
    /// Tool call id for tool role messages (OpenAI-style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name — set on assistant tool-invocation messages so clients can reconstruct
    /// the provider-specific tool_calls array when replaying history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

impl Message {
    /// Create a new text message
    pub fn text<S: Into<String>>(role: Role, text: S) -> Self {
        Self {
            role,
            content: vec![ContentBlock::Text { text: text.into() }],
            tool_call_id: None,
            tool_name: None,
        }
    }

    /// Create a user message with text content
    pub fn user<S: Into<String>>(text: S) -> Self {
        Self::text(Role::User, text)
    }

    /// Create an assistant message with text content
    pub fn assistant<S: Into<String>>(text: S) -> Self {
        Self::text(Role::Assistant, text)
    }

    /// Create a system message with text content
    pub fn system<S: Into<String>>(text: S) -> Self {
        Self::text(Role::System, text)
    }

    /// Create a tool-result message with text content
    pub fn tool<S: Into<String>>(tool_call_id: impl Into<String>, text: S) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::Text { text: text.into() }],
            tool_call_id: Some(tool_call_id.into()),
            tool_name: None,
        }
    }
}

/// Token usage information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the input prompt
    pub input_tokens: u32,
    /// Number of tokens in the output completion
    pub output_tokens: u32,
}

/// Response format type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Plain text response
    Text,
    /// JSON object response
    JsonObject,
}

/// Generic completion request (provider-agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Messages for the conversation
    pub messages: Vec<Message>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Model to use (provider-specific)
    pub model: String,
    /// Optional system message
    pub system: Option<String>,
    /// Temperature for randomness (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Top-p sampling parameter
    pub top_p: Option<f32>,
    /// Stop sequences
    pub stop_sequences: Option<Vec<String>>,
    /// Tools available to the LLM
    pub tools: Option<Vec<crate::tools::Tool>>,
    /// Tool choice strategy
    pub tool_choice: Option<crate::tools::ToolChoice>,
    /// Response format (text or JSON object)
    pub response_format: Option<ResponseFormat>,
}

/// Generic completion response (provider-agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Generated content
    pub content: Vec<ContentBlock>,
    /// Role of the response
    pub role: Role,
    /// Token usage information
    pub usage: Usage,
    /// Stop reason
    pub stop_reason: Option<String>,
    /// Tool calls requested by the LLM
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
}

/// Streaming response chunk
#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    /// Text content in this chunk
    pub content: String,
    /// Whether this is the final chunk
    pub is_finished: bool,
    /// Tool calls (if any)
    pub tool_calls: Vec<crate::tools::ToolCall>,
}
