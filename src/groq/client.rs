use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

use crate::{
    error::LlmError,
    groq::types::{
        GroqChatCompletionRequest, GroqChatCompletionResponse, GroqErrorResponse,
        GroqResponseFormat,
    },
};

pub struct GroqClient {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl GroqClient {
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
            base_url: "https://api.groq.com/openai".to_string(),
            http_client,
        })
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub async fn create_chat_completion(
        &self,
        request: GroqChatCompletionRequest,
    ) -> Result<GroqChatCompletionResponse, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LlmError::authentication("Invalid API key format"))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

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
            let groq_response: GroqChatCompletionResponse = response
                .json()
                .await
                .map_err(|e| LlmError::internal(format!("Failed to parse response: {}", e)))?;
            Ok(groq_response)
        } else {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            if let Ok(error_response) = serde_json::from_str::<GroqErrorResponse>(&error_text) {
                match status {
                    reqwest::StatusCode::UNAUTHORIZED => {
                        Err(LlmError::authentication(error_response.error.message))
                    }
                    reqwest::StatusCode::FORBIDDEN => {
                        Err(LlmError::authentication(error_response.error.message))
                    }
                    reqwest::StatusCode::BAD_REQUEST => {
                        Err(LlmError::invalid_request(error_response.error.message))
                    }
                    reqwest::StatusCode::NOT_FOUND => {
                        Err(LlmError::api_error(404, error_response.error.message))
                    }
                    reqwest::StatusCode::PAYLOAD_TOO_LARGE => {
                        Err(LlmError::invalid_request("Request too large"))
                    }
                    reqwest::StatusCode::TOO_MANY_REQUESTS => {
                        Err(LlmError::rate_limit(error_response.error.message, None))
                    }
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                        Err(LlmError::api_error(500, error_response.error.message))
                    }
                    _ => Err(LlmError::api_error(
                        status.as_u16(),
                        error_response.error.message,
                    )),
                }
            } else {
                match status {
                    reqwest::StatusCode::UNAUTHORIZED => Err(LlmError::authentication(error_text)),
                    reqwest::StatusCode::FORBIDDEN => Err(LlmError::authentication(error_text)),
                    reqwest::StatusCode::BAD_REQUEST => Err(LlmError::invalid_request(error_text)),
                    reqwest::StatusCode::NOT_FOUND => Err(LlmError::api_error(404, error_text)),
                    reqwest::StatusCode::PAYLOAD_TOO_LARGE => {
                        Err(LlmError::invalid_request("Request too large"))
                    }
                    reqwest::StatusCode::TOO_MANY_REQUESTS => {
                        Err(LlmError::rate_limit(error_text, None))
                    }
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR => {
                        Err(LlmError::api_error(500, error_text))
                    }
                    _ => Err(LlmError::api_error(status.as_u16(), error_text)),
                }
            }
        }
    }
}

impl GroqChatCompletionResponse {
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
impl crate::client::LlmClient for GroqClient {
    async fn complete(
        &self,
        request: crate::types::CompletionRequest,
    ) -> Result<crate::types::CompletionResponse, LlmError> {
        let groq_messages = request
            .messages
            .into_iter()
            .map(|msg| {
                let role = match msg.role {
                    crate::types::Role::User => crate::groq::types::GroqRole::User,
                    crate::types::Role::Assistant => crate::groq::types::GroqRole::Assistant,
                    crate::types::Role::System => crate::groq::types::GroqRole::System,
                    crate::types::Role::Tool => crate::groq::types::GroqRole::Tool,
                };

                let content = msg
                    .content
                    .into_iter()
                    .map(|block| match block {
                        crate::types::ContentBlock::Text { text } => Ok(text),
                        crate::types::ContentBlock::Image { .. } => Err(
                            LlmError::invalid_request("Image content not supported"),
                        ),
                    })
                    .collect::<Result<Vec<String>, LlmError>>()?
                    .join("");

                Ok(crate::groq::types::GroqMessage {
                    role,
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                })
            })
            .collect::<Result<Vec<crate::groq::types::GroqMessage>, LlmError>>()?;

        let response_format = request.response_format.map(|rf| match rf {
            crate::types::ResponseFormat::Text => GroqResponseFormat::text(),
            crate::types::ResponseFormat::JsonObject => GroqResponseFormat::json_object(),
        });

        let groq_request = GroqChatCompletionRequest {
            model: request.model,
            messages: groq_messages,
            max_completion_tokens: Some(request.max_tokens),
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences,
            stream: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format,
        };

        let groq_response = self.create_chat_completion(groq_request).await?;

        if groq_response.choices.is_empty() {
            return Err(LlmError::internal("No completion choices returned"));
        }

        let choice = &groq_response.choices[0];
        let content = vec![crate::types::ContentBlock::Text {
            text: choice.message.content.clone(),
        }];

        Ok(crate::types::CompletionResponse {
            content,
            role: match choice.message.role {
                crate::groq::types::GroqRole::User => crate::types::Role::User,
                crate::groq::types::GroqRole::Assistant => crate::types::Role::Assistant,
                crate::groq::types::GroqRole::System => crate::types::Role::System,
                crate::groq::types::GroqRole::Tool => crate::types::Role::Tool,
            },
            usage: crate::types::Usage {
                input_tokens: groq_response
                    .usage
                    .as_ref()
                    .map(|u| u.prompt_tokens)
                    .unwrap_or(0),
                output_tokens: groq_response
                    .usage
                    .as_ref()
                    .map(|u| u.completion_tokens)
                    .unwrap_or(0),
            },
            stop_reason: choice.finish_reason.clone(),
            tool_calls: None,
        })
    }

    fn provider_name(&self) -> &str {
        crate::providers::GROQ
    }

    fn model_name(&self) -> &str {
        crate::models::groq::GPT_OSS_20B_ID
    }
}
