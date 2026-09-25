use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

use crate::{
    error::LlmError,
    openai::types::OpenAIFunctionCall,
    openrouter::{
        tools::OpenRouterToolFormat,
        types::{
            OpenRouterChatCompletionRequest, OpenRouterChatCompletionResponse,
            OpenRouterChatCompletionResult, OpenRouterErrorResponse, OpenRouterMessage,
            OpenRouterModelInfo, OpenRouterModelsResponse, OpenRouterProviderPreferences,
            OpenRouterResponseFormat, OpenRouterRole, OpenRouterToolCall,
        },
    },
    tools::ProviderToolFormat,
};

/// OpenRouter client.
///
/// OpenRouter proxies many models via an OpenAI-compatible Chat Completions API.
/// Free programming models are not hardcoded — call [`list_free_programming_models`]
/// to discover them at runtime, then set one with [`with_model`] before completing.
///
/// ```rust,no_run
/// # tokio_test::block_on(async {
/// use nocodo_llm_sdk::openrouter::OpenRouterClient;
///
/// let client = OpenRouterClient::new("your-api-key")?;
/// let models = client.list_free_programming_models().await?;
/// let client = client.with_model(&models[0].id);
/// # Ok::<(), nocodo_llm_sdk::error::LlmError>(())
/// # });
/// ```
///
/// [`list_free_programming_models`]: OpenRouterClient::list_free_programming_models
/// [`with_model`]: OpenRouterClient::with_model
#[derive(Debug)]
pub struct OpenRouterClient {
    api_key: String,
    model: Option<String>,
    base_url: String,
    http_client: reqwest::Client,
    provider_preferences: Option<OpenRouterProviderPreferences>,
    response_format: Option<OpenRouterResponseFormat>,
}

impl OpenRouterClient {
    pub fn new(api_key: impl Into<String>) -> Result<Self, LlmError> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(LlmError::authentication("API key cannot be empty"));
        }

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| LlmError::Network { source: e })?;

        Ok(Self {
            api_key,
            model: None,
            base_url: "https://openrouter.ai/api".to_string(),
            http_client,
            provider_preferences: None,
            response_format: None,
        })
    }

    /// Set the model to use for chat completions.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set provider-routing preferences used by [`crate::client::LlmClient::complete`].
    pub fn with_provider_preferences(mut self, preferences: OpenRouterProviderPreferences) -> Self {
        self.provider_preferences = Some(preferences);
        self
    }

    /// Set the response format used by [`crate::client::LlmClient::complete`].
    ///
    /// Takes precedence over [`crate::types::CompletionRequest::response_format`],
    /// which cannot express a strict JSON schema. Use it with
    /// [`OpenRouterResponseFormat::json_schema`] for structured output.
    pub fn with_response_format(mut self, format: OpenRouterResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    fn auth_headers(&self) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LlmError::authentication("Invalid API key format"))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    /// Return all free-tier programming models available on OpenRouter.
    ///
    /// Queries `GET /v1/models?category=programming` and filters to models where
    /// both prompt and completion pricing are `"0"`.
    pub async fn list_free_programming_models(&self) -> Result<Vec<OpenRouterModelInfo>, LlmError> {
        let url = format!("{}/v1/models?category=programming", self.base_url);
        let headers = self.auth_headers()?;

        let response = self
            .http_client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| LlmError::Network { source: e })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::api_error(status.as_u16(), body));
        }

        let models_response: OpenRouterModelsResponse = response
            .json()
            .await
            .map_err(|e| LlmError::internal(format!("Failed to parse models response: {}", e)))?;

        Ok(models_response
            .data
            .into_iter()
            .filter(|m| m.pricing.is_free())
            .collect())
    }

    pub async fn create_chat_completion(
        &self,
        request: OpenRouterChatCompletionRequest,
    ) -> Result<OpenRouterChatCompletionResponse, LlmError> {
        Ok(self
            .create_chat_completion_with_metadata(request)
            .await?
            .response)
    }

    /// Create a completion and retain the exact successful response bytes.
    pub async fn create_chat_completion_with_metadata(
        &self,
        request: OpenRouterChatCompletionRequest,
    ) -> Result<OpenRouterChatCompletionResult, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let headers = self.auth_headers()?;

        let response = self
            .http_client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LlmError::Network { source: e })?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| LlmError::Network { source: e })?;
        if status.is_success() {
            let resp: OpenRouterChatCompletionResponse = serde_json::from_slice(&body)
                .map_err(|e| LlmError::internal(format!("Failed to parse response: {}", e)))?;
            return Ok(OpenRouterChatCompletionResult {
                response: resp,
                raw_response: body.to_vec(),
            });
        }

        let error_text = String::from_utf8_lossy(&body).into_owned();
        if let Ok(err) = serde_json::from_str::<OpenRouterErrorResponse>(&error_text) {
            let error_type = err
                .error
                .metadata
                .and_then(|metadata| metadata.error_type)
                .unwrap_or_else(|| "unknown".to_string());
            Err(LlmError::openrouter_api_error(status.as_u16(), error_type))
        } else {
            match status {
                reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                    Err(LlmError::authentication(error_text))
                }
                reqwest::StatusCode::BAD_REQUEST => Err(LlmError::invalid_request(error_text)),
                reqwest::StatusCode::TOO_MANY_REQUESTS => {
                    Err(LlmError::rate_limit(error_text, None))
                }
                _ => Err(LlmError::api_error(status.as_u16(), error_text)),
            }
        }
    }
}

impl OpenRouterChatCompletionResponse {
    pub fn tool_calls(&self) -> Option<Vec<crate::tools::ToolCall>> {
        self.choices
            .first()?
            .message
            .tool_calls
            .as_ref()
            .map(|calls| {
                calls
                    .iter()
                    .map(|call| {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&call.function.arguments)
                                .unwrap_or(serde_json::Value::Null);
                        crate::tools::ToolCall::new(
                            call.id.clone(),
                            call.function.name.clone(),
                            arguments,
                        )
                    })
                    .collect()
            })
    }
}

#[async_trait]
impl crate::client::LlmClient for OpenRouterClient {
    async fn complete(
        &self,
        request: crate::types::CompletionRequest,
    ) -> Result<crate::types::CompletionResponse, LlmError> {
        let model = self
            .model
            .clone()
            .ok_or_else(|| LlmError::invalid_request("No model set — call with_model() first"))?;

        let mut messages: Vec<OpenRouterMessage> = Vec::new();

        if let Some(system) = request.system {
            messages.push(OpenRouterMessage::system(system));
        }

        for msg in request.messages {
            let content = msg
                .content
                .into_iter()
                .map(|block| match block {
                    crate::types::ContentBlock::Text { text } => Ok(text),
                    crate::types::ContentBlock::Image { .. } => {
                        Err(LlmError::invalid_request("Image content not supported"))
                    }
                })
                .collect::<Result<Vec<String>, LlmError>>()?
                .join("");

            let or_msg = match msg.role {
                crate::types::Role::Assistant if msg.tool_call_id.is_some() => {
                    let call_id = msg.tool_call_id.unwrap();
                    let tool_name = msg.tool_name.unwrap_or_default();
                    OpenRouterMessage {
                        role: OpenRouterRole::Assistant,
                        content: String::new(),
                        tool_calls: Some(vec![OpenRouterToolCall {
                            id: call_id,
                            r#type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: tool_name,
                                arguments: content,
                            },
                        }]),
                        tool_call_id: None,
                    }
                }
                crate::types::Role::Tool => OpenRouterMessage {
                    role: OpenRouterRole::Tool,
                    content,
                    tool_calls: None,
                    tool_call_id: msg.tool_call_id,
                },
                role => {
                    let or_role = match role {
                        crate::types::Role::User => OpenRouterRole::User,
                        crate::types::Role::Assistant => OpenRouterRole::Assistant,
                        crate::types::Role::System => OpenRouterRole::System,
                        crate::types::Role::Tool => OpenRouterRole::Tool,
                    };
                    OpenRouterMessage::new(or_role, content)
                }
            };
            messages.push(or_msg);
        }

        let tools = request.tools.map(|tools| {
            tools
                .iter()
                .map(|t| OpenRouterToolFormat::to_provider_tool(t))
                .collect()
        });

        let tool_choice = request
            .tool_choice
            .map(|c| OpenRouterToolFormat::to_provider_tool_choice(&c));

        let response_format = self.response_format.clone().or_else(|| {
            request.response_format.map(|rf| match rf {
                crate::types::ResponseFormat::Text => OpenRouterResponseFormat::text(),
                crate::types::ResponseFormat::JsonObject => OpenRouterResponseFormat::json_object(),
            })
        });

        let or_request = OpenRouterChatCompletionRequest {
            model,
            messages,
            max_completion_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences,
            stream: None,
            tools,
            tool_choice,
            response_format,
            provider: self.provider_preferences.clone(),
            reasoning: None,
        };

        let or_response = self.create_chat_completion(or_request).await?;

        if or_response.choices.is_empty() {
            return Err(LlmError::internal("No completion choices returned"));
        }

        let choice = &or_response.choices[0];
        let tool_calls = or_response.tool_calls();
        let tool_calls = match tool_calls {
            Some(tc) if !tc.is_empty() => Some(tc),
            _ => None,
        };

        Ok(crate::types::CompletionResponse {
            content: vec![crate::types::ContentBlock::Text {
                text: choice.message.content.clone(),
            }],
            role: match choice.message.role {
                OpenRouterRole::User => crate::types::Role::User,
                OpenRouterRole::Assistant => crate::types::Role::Assistant,
                OpenRouterRole::System => crate::types::Role::System,
                OpenRouterRole::Tool => crate::types::Role::Tool,
            },
            usage: crate::types::Usage {
                input_tokens: or_response
                    .usage
                    .as_ref()
                    .map(|u| u.prompt_tokens)
                    .unwrap_or(0),
                output_tokens: or_response
                    .usage
                    .as_ref()
                    .map(|u| u.completion_tokens)
                    .unwrap_or(0),
                reasoning_tokens: None,
            },
            stop_reason: choice.finish_reason.clone(),
            tool_calls,
        })
    }

    fn provider_name(&self) -> &str {
        crate::providers::OPENROUTER
    }

    fn model_name(&self) -> &str {
        self.model.as_deref().unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn metadata_completion_preserves_raw_body_and_routing_identity() {
        let mut server = mockito::Server::new_async().await;
        let raw = r#"{"id":"generation-1","created":123,"model":"author/model","provider":"Provider A","choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}"#;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(raw)
            .create_async()
            .await;
        let client = OpenRouterClient::new("test-key")
            .unwrap()
            .with_base_url(server.url());
        let request = OpenRouterChatCompletionRequest {
            model: "author/model".into(),
            messages: vec![OpenRouterMessage::user("hello")],
            max_completion_tokens: Some(64),
            temperature: Some(0.0),
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: Some(OpenRouterResponseFormat::json_object()),
            provider: Some(OpenRouterProviderPreferences {
                order: Some(vec!["Provider A".into()]),
                allow_fallbacks: Some(false),
                require_parameters: Some(true),
                data_collection: Some(crate::openrouter::types::OpenRouterDataCollection::Deny),
                zdr: Some(true),
            }),
            reasoning: None,
        };

        let result = client
            .create_chat_completion_with_metadata(request)
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(result.response.model, "author/model");
        assert_eq!(result.response.provider.as_deref(), Some("Provider A"));
        assert_eq!(result.raw_response, raw.as_bytes());
    }

    #[tokio::test]
    async fn client_response_format_sends_strict_json_schema_on_the_wire() {
        use crate::client::LlmClient;

        let mut server = mockito::Server::new_async().await;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        });
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "task_zero_answer",
                        "strict": true,
                        "schema": schema
                    }
                }
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{"id":"generation-1","created":123,"model":"author/model","choices":[{"index":0,"message":{"role":"assistant","content":"{\"answer\":\"ok\"}"},"finish_reason":"stop"}]}"#,
            )
            .create_async()
            .await;

        let client = OpenRouterClient::new("test-key")
            .unwrap()
            .with_base_url(server.url())
            .with_model("author/model")
            .with_response_format(OpenRouterResponseFormat::json_schema(
                "task_zero_answer",
                schema.clone(),
            ));

        let request = crate::types::CompletionRequest {
            messages: vec![crate::types::Message {
                role: crate::types::Role::User,
                content: vec![crate::types::ContentBlock::Text {
                    text: "hello".into(),
                }],
                tool_call_id: None,
                tool_name: None,
            }],
            max_tokens: 64,
            model: "author/model".into(),
            system: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        client.complete(request).await.unwrap();
        mock.assert_async().await;
    }

    #[test]
    fn openrouter_error_retains_only_status_and_canonical_error_type() {
        let parsed: OpenRouterErrorResponse = serde_json::from_str(
            r#"{"error":{"code":503,"message":"sensitive provider detail","metadata":{"error_type":"provider_overloaded","provider_code":"private"}}}"#,
        )
        .unwrap();
        let error = LlmError::openrouter_api_error(
            503,
            parsed
                .error
                .metadata
                .and_then(|metadata| metadata.error_type)
                .unwrap(),
        );
        assert_eq!(
            error.openrouter_diagnostic(),
            Some((503, "provider_overloaded"))
        );
        assert!(!error.to_string().contains("sensitive provider detail"));
        assert!(!error.to_string().contains("private"));
    }
}
