use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

use crate::{
    error::LlmError,
    openai::types::OpenAIFunctionCall,
    openrouter::{
        tools::OpenRouterToolFormat,
        types::{
            OpenRouterChatCompletionRequest, OpenRouterChatCompletionResponse,
            OpenRouterErrorResponse, OpenRouterMessage, OpenRouterModelInfo,
            OpenRouterModelsResponse, OpenRouterResponseFormat, OpenRouterRole, OpenRouterToolCall,
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
        if status.is_success() {
            let resp: OpenRouterChatCompletionResponse = response
                .json()
                .await
                .map_err(|e| LlmError::internal(format!("Failed to parse response: {}", e)))?;
            return Ok(resp);
        }

        let error_text = response.text().await.unwrap_or_default();
        if let Ok(err) = serde_json::from_str::<OpenRouterErrorResponse>(&error_text) {
            match status {
                reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                    Err(LlmError::authentication(err.error.message))
                }
                reqwest::StatusCode::BAD_REQUEST => {
                    Err(LlmError::invalid_request(err.error.message))
                }
                reqwest::StatusCode::TOO_MANY_REQUESTS => {
                    Err(LlmError::rate_limit(err.error.message, None))
                }
                _ => Err(LlmError::api_error(status.as_u16(), err.error.message)),
            }
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
        self.choices.first()?.message.tool_calls.as_ref().map(|calls| {
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

        let response_format = request.response_format.map(|rf| match rf {
            crate::types::ResponseFormat::Text => OpenRouterResponseFormat::text(),
            crate::types::ResponseFormat::JsonObject => OpenRouterResponseFormat::json_object(),
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
