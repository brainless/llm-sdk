use crate::{
    error::LlmError,
    groq::{
        client::GroqClient,
        tools::GroqToolFormat,
        types::{
            GroqChatCompletionRequest, GroqChatCompletionResponse, GroqMessage, GroqResponseFormat,
            GroqRole, GroqTool,
        },
    },
    tools::{ProviderToolFormat, Tool, ToolChoice, ToolResult},
};

pub struct GroqMessageBuilder<'a> {
    client: &'a GroqClient,
    model: Option<String>,
    max_tokens: Option<u32>,
    messages: Vec<GroqMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    stream: Option<bool>,
    reasoning_effort: Option<String>,
    tools: Option<Vec<GroqTool>>,
    tool_choice: Option<serde_json::Value>,
    response_format: Option<GroqResponseFormat>,
}

impl<'a> GroqMessageBuilder<'a> {
    pub fn new(client: &'a GroqClient) -> Self {
        Self {
            client,
            model: None,
            max_tokens: None,
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Add a message. Valid roles: "system", "user", "assistant", "tool".
    pub fn message(mut self, role: impl Into<String>, content: impl Into<String>) -> Self {
        let role_str = role.into();
        let role = match role_str.as_str() {
            "system" => GroqRole::System,
            "user" => GroqRole::User,
            "assistant" => GroqRole::Assistant,
            "tool" => GroqRole::Tool,
            _ => {
                tracing::warn!("Invalid role '{}', defaulting to 'user'", role_str);
                GroqRole::User
            }
        };
        self.messages.push(GroqMessage::new(role, content));
        self
    }

    pub fn system_message(self, content: impl Into<String>) -> Self {
        self.message("system", content)
    }

    pub fn user_message(self, content: impl Into<String>) -> Self {
        self.message("user", content)
    }

    pub fn assistant_message(self, content: impl Into<String>) -> Self {
        self.message("assistant", content)
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop = Some(stop_sequences);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Set reasoning effort for gpt-oss models. Values: "low", "medium" (default), "high".
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        let tools = self.tools.get_or_insert_with(Vec::new);
        tools.push(GroqToolFormat::to_provider_tool(&tool));
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        for tool in tools {
            self = self.tool(tool);
        }
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(GroqToolFormat::to_provider_tool_choice(&choice));
        self
    }

    pub fn response_format(mut self, format: GroqResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    pub fn tool_result(mut self, result: ToolResult) -> Self {
        self.messages.push(GroqMessage::tool_result(
            result.tool_call_id(),
            result.content(),
        ));
        self
    }

    pub async fn send(self) -> Result<GroqChatCompletionResponse, LlmError> {
        let request = GroqChatCompletionRequest {
            model: self
                .model
                .ok_or_else(|| LlmError::invalid_request("Model must be specified"))?,
            messages: self.messages,
            max_completion_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop,
            stream: self.stream,
            reasoning_effort: self.reasoning_effort,
            tools: self.tools,
            tool_choice: self.tool_choice,
            response_format: self.response_format,
        };

        self.client.create_chat_completion(request).await
    }
}

impl GroqClient {
    pub fn message_builder(&self) -> GroqMessageBuilder<'_> {
        GroqMessageBuilder::new(self)
    }
}
