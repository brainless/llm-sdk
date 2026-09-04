use crate::{
    error::LlmError,
    mixlayer::{
        client::MixlayerClient,
        tools::MixlayerToolFormat,
        types::{
            MixlayerChatCompletionRequest, MixlayerChatCompletionResponse, MixlayerMessage,
            MixlayerResponseFormat, MixlayerTool, MixlayerWebSearchOptions,
        },
    },
    tools::{ProviderToolFormat, Tool, ToolChoice, ToolResult},
};

/// Builder for a MixLayer Chat Completions request.
pub struct MixlayerMessageBuilder<'a> {
    client: &'a MixlayerClient,
    model: Option<String>,
    messages: Vec<MixlayerMessage>,
    max_completion_tokens: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    seed: Option<i64>,
    reasoning_effort: Option<String>,
    thinking: Option<bool>,
    response_format: Option<MixlayerResponseFormat>,
    tools: Option<Vec<MixlayerTool>>,
    tool_choice: Option<serde_json::Value>,
    web_search_options: Option<MixlayerWebSearchOptions>,
    metadata: Option<serde_json::Value>,
    store: Option<bool>,
}

impl<'a> MixlayerMessageBuilder<'a> {
    pub fn new(client: &'a MixlayerClient) -> Self {
        Self {
            client,
            model: None,
            messages: Vec::new(),
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
            thinking: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            web_search_options: None,
            metadata: None,
            store: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn message(
        mut self,
        role: crate::mixlayer::MixlayerRole,
        content: impl Into<String>,
    ) -> Self {
        self.messages.push(MixlayerMessage::new(role, content));
        self
    }

    pub fn system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(MixlayerMessage::system(content));
        self
    }

    pub fn user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(MixlayerMessage::user(content));
        self
    }

    pub fn assistant_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(MixlayerMessage::assistant(content));
        self
    }

    pub fn tool_result(mut self, result: ToolResult) -> Self {
        self.messages.push(MixlayerMessage::tool_result(
            result.tool_call_id(),
            result.content(),
        ));
        self
    }

    pub fn max_completion_tokens(mut self, tokens: u32) -> Self {
        self.max_completion_tokens = Some(tokens);
        self
    }

    /// Set the legacy `max_tokens` field. Prefer [`Self::max_completion_tokens`].
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    pub fn temperature(mut self, value: f32) -> Self {
        self.temperature = Some(value);
        self
    }
    pub fn top_p(mut self, value: f32) -> Self {
        self.top_p = Some(value);
        self
    }
    pub fn top_k(mut self, value: u32) -> Self {
        self.top_k = Some(value);
        self
    }
    pub fn frequency_penalty(mut self, value: f32) -> Self {
        self.frequency_penalty = Some(value);
        self
    }
    pub fn presence_penalty(mut self, value: f32) -> Self {
        self.presence_penalty = Some(value);
        self
    }
    pub fn repetition_penalty(mut self, value: f32) -> Self {
        self.repetition_penalty = Some(value);
        self
    }
    pub fn stop_sequences(mut self, value: Vec<String>) -> Self {
        self.stop = Some(value);
        self
    }
    pub fn seed(mut self, value: i64) -> Self {
        self.seed = Some(value);
        self
    }
    pub fn reasoning_effort(mut self, value: impl Into<String>) -> Self {
        self.reasoning_effort = Some(value.into());
        self
    }
    pub fn thinking(mut self, value: bool) -> Self {
        self.thinking = Some(value);
        self
    }
    pub fn response_format(mut self, value: MixlayerResponseFormat) -> Self {
        self.response_format = Some(value);
        self
    }
    pub fn json_schema_response_format(self, schema: serde_json::Value) -> Self {
        self.response_format(MixlayerResponseFormat::json_schema(schema))
    }
    pub fn web_search_options(mut self, value: MixlayerWebSearchOptions) -> Self {
        self.web_search_options = Some(value);
        self
    }
    pub fn metadata(mut self, value: serde_json::Value) -> Self {
        self.metadata = Some(value);
        self
    }
    pub fn store(mut self, value: bool) -> Self {
        self.store = Some(value);
        self
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools
            .get_or_insert_with(Vec::new)
            .push(MixlayerToolFormat::to_provider_tool(&tool));
        self
    }

    /// Add a function tool and request strict argument-schema enforcement.
    pub fn strict_tool(mut self, tool: Tool) -> Self {
        let mut provider_tool = MixlayerToolFormat::to_provider_tool(&tool);
        provider_tool.function.strict = Some(true);
        self.tools.get_or_insert_with(Vec::new).push(provider_tool);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        for tool in tools {
            self = self.tool(tool);
        }
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(MixlayerToolFormat::to_provider_tool_choice(&choice));
        self
    }

    pub async fn send(self) -> Result<MixlayerChatCompletionResponse, LlmError> {
        let request = MixlayerChatCompletionRequest {
            model: self
                .model
                .unwrap_or_else(|| self.client.model_name().to_string()),
            messages: self.messages,
            max_completion_tokens: self.max_completion_tokens,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_penalty: self.repetition_penalty,
            stop: self.stop,
            seed: self.seed,
            reasoning_effort: self.reasoning_effort,
            thinking: self.thinking,
            response_format: self.response_format,
            tools: self.tools,
            tool_choice: self.tool_choice,
            web_search_options: self.web_search_options,
            metadata: self.metadata,
            store: self.store,
        };
        self.client.create_chat_completion(request).await
    }
}
