use crate::{
    error::LlmError,
    llama_cpp::{
        client::LlamaCppClient,
        tools::LlamaCppToolFormat,
        types::{LlamaCppChatCompletionRequest, LlamaCppMessage, LlamaCppRole},
    },
    tools::{ProviderToolFormat, Tool, ToolChoice},
};

/// Builder for creating llama.cpp chat completion requests
pub struct LlamaCppMessageBuilder<'a> {
    client: &'a LlamaCppClient,
    model: Option<String>,
    max_tokens: Option<u32>,
    messages: Vec<LlamaCppMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    stream: Option<bool>,
    tools: Option<Vec<crate::openai::types::OpenAITool>>,
    tool_choice: Option<ToolChoice>,
    parallel_tool_calls: Option<bool>,
    response_format: Option<serde_json::Value>,
}

impl<'a> LlamaCppMessageBuilder<'a> {
    /// Create a new message builder
    pub fn new(client: &'a LlamaCppClient) -> Self {
        Self {
            client,
            model: None,
            max_tokens: None,
            messages: Vec::new(),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            response_format: None,
        }
    }

    /// Set the model to use
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the maximum number of tokens to generate
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Add a message to the conversation
    pub fn message(mut self, role: impl Into<String>, content: impl Into<String>) -> Self {
        let role_str = role.into();
        let role = match role_str.as_str() {
            "system" => LlamaCppRole::System,
            "user" => LlamaCppRole::User,
            "assistant" => LlamaCppRole::Assistant,
            "tool" => LlamaCppRole::Tool,
            _ => {
                tracing::warn!("Invalid role '{}', defaulting to 'user'", role_str);
                LlamaCppRole::User
            }
        };

        self.messages.push(LlamaCppMessage::new(role, content));
        self
    }

    /// Add a system message
    pub fn system_message(self, content: impl Into<String>) -> Self {
        self.message("system", content)
    }

    /// Add a user message
    pub fn user_message(self, content: impl Into<String>) -> Self {
        self.message("user", content)
    }

    /// Add an assistant message
    pub fn assistant_message(self, content: impl Into<String>) -> Self {
        self.message("assistant", content)
    }

    /// Add a tool message
    pub fn tool_message(self, content: impl Into<String>) -> Self {
        self.message("tool", content)
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set top-p sampling parameter
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn stop_sequences(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Enable or disable streaming
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Add a tool to the request
    pub fn tool(mut self, tool: Tool) -> Self {
        let tools = self.tools.get_or_insert_with(Vec::new);
        tools.push(LlamaCppToolFormat::to_provider_tool(&tool));
        self
    }

    /// Add multiple tools to the request
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        for tool in tools {
            self = self.tool(tool);
        }
        self
    }

    /// Set tool choice strategy
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Enable or disable parallel tool calls
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.parallel_tool_calls = Some(enabled);
        self
    }

    /// Request grammar-constrained JSON output matching `schema`, sent in
    /// mlxcel's documented OpenAI-compatible wire shape:
    /// `{"type":"json_schema","json_schema":{"name","strict","schema"}}`.
    ///
    /// `schema` is the raw JSON Schema document (e.g. `Tool::parameters()`
    /// serialized to `serde_json::Value`). `strict` is forwarded as-is;
    /// mlxcel's structured-output extraction (`extract_json_schema_from_response_format`)
    /// currently ignores `name`/`strict` and reads only `schema`, but both
    /// are still sent for wire compatibility with the documented
    /// OpenAI Chat Completions shape and any future stricter validation.
    pub fn response_format_json_schema(
        mut self,
        name: impl Into<String>,
        schema: serde_json::Value,
        strict: bool,
    ) -> Self {
        self.response_format = Some(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": name.into(),
                "strict": strict,
                "schema": schema,
            }
        }));
        self
    }

    /// Send the request and get the response
    pub async fn send(
        self,
    ) -> Result<crate::llama_cpp::types::LlamaCppChatCompletionResponse, LlmError> {
        let client = self.client;
        let request = self.build_request()?;
        client.create_chat_completion(request).await
    }

    fn build_request(self) -> Result<LlamaCppChatCompletionRequest, LlmError> {
        Ok(LlamaCppChatCompletionRequest {
            model: self
                .model
                .ok_or_else(|| LlmError::invalid_request("Model must be specified"))?,
            messages: self.messages,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop,
            stream: self.stream,
            tools: self.tools,
            tool_choice: self
                .tool_choice
                .as_ref()
                .map(LlamaCppToolFormat::to_provider_tool_choice),
            parallel_tool_calls: self.parallel_tool_calls,
            response_format: self.response_format,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug, Serialize, Deserialize, JsonSchema)]
    struct PlanSubmissionParams {
        summary: String,
    }

    #[test]
    fn forwards_specific_tool_choice_and_parallel_tool_calls_into_request_body() {
        let client = LlamaCppClient::new().unwrap();
        let tool = Tool::from_type::<PlanSubmissionParams>()
            .name("submit_plan")
            .description("Submit the task plan")
            .build();

        let request = LlamaCppMessageBuilder::new(&client)
            .model("qwen-test")
            .user_message("plan the tasks")
            .tool(tool)
            .tool_choice(ToolChoice::Specific {
                name: "submit_plan".to_string(),
            })
            .parallel_tool_calls(false)
            .build_request()
            .unwrap();

        let body = serde_json::to_value(&request).unwrap();
        assert_eq!(
            body["tool_choice"],
            json!({
                "type": "function",
                "function": { "name": "submit_plan" }
            })
        );
        assert_eq!(body["parallel_tool_calls"], json!(false));
    }

    #[test]
    fn omits_unset_tool_fields_from_request_body() {
        let client = LlamaCppClient::new().unwrap();
        let request = LlamaCppMessageBuilder::new(&client)
            .model("qwen-test")
            .user_message("hello")
            .build_request()
            .unwrap();

        let body = serde_json::to_value(&request).unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn response_format_json_schema_reaches_the_outbound_body_byte_exact() {
        let client = LlamaCppClient::new().unwrap();
        let schema = Tool::from_type::<PlanSubmissionParams>()
            .name("submit_plan")
            .description("Submit the task plan")
            .build()
            .parameters()
            .clone();
        let schema_value = serde_json::to_value(&schema).unwrap();

        let request = LlamaCppMessageBuilder::new(&client)
            .model("qwen-test")
            .user_message("plan the tasks")
            .response_format_json_schema("submit_plan", schema_value.clone(), true)
            .build_request()
            .unwrap();

        let body = serde_json::to_value(&request).unwrap();
        assert_eq!(
            body["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "submit_plan",
                    "strict": true,
                    "schema": schema_value,
                }
            })
        );
        // mlxcel's own documented wire shape has exactly these three keys
        // inside json_schema, nothing more.
        let inner = body["response_format"]["json_schema"].as_object().unwrap();
        let mut keys: Vec<&str> = inner.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["name", "schema", "strict"]);
    }

    #[test]
    fn response_format_json_schema_forwards_strict_false_unchanged() {
        let client = LlamaCppClient::new().unwrap();
        let request = LlamaCppMessageBuilder::new(&client)
            .model("qwen-test")
            .user_message("plan the tasks")
            .response_format_json_schema("submit_plan", json!({"type": "object"}), false)
            .build_request()
            .unwrap();

        let body = serde_json::to_value(&request).unwrap();
        assert_eq!(
            body["response_format"]["json_schema"]["strict"],
            json!(false)
        );
    }
}
