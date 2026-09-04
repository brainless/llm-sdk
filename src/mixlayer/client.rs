use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

use crate::{
    error::LlmError,
    mixlayer::{
        tools::MixlayerToolFormat,
        types::{
            MixlayerChatCompletionRequest, MixlayerChatCompletionResponse, MixlayerFunctionCall,
            MixlayerMessage, MixlayerResponseFormat, MixlayerRole, MixlayerToolCall,
        },
    },
    tools::ProviderToolFormat,
};

/// Client for MixLayer's OpenAI-compatible Chat Completions API.
#[derive(Debug)]
pub struct MixlayerClient {
    api_key: String,
    model: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl MixlayerClient {
    pub fn new(api_key: impl Into<String>) -> Result<Self, LlmError> {
        let api_key = api_key.into();
        if api_key.is_empty() {
            return Err(LlmError::authentication("API key cannot be empty"));
        }
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|source| LlmError::Network { source })?;
        Ok(Self {
            api_key,
            model: crate::models::mixlayer::QWEN_3_5_4B_FREE_ID.to_string(),
            base_url: "https://models.mixlayer.ai".to_string(),
            http_client,
        })
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn message_builder(&self) -> crate::mixlayer::builder::MixlayerMessageBuilder<'_> {
        crate::mixlayer::builder::MixlayerMessageBuilder::new(self)
    }

    pub async fn create_chat_completion(
        &self,
        request: MixlayerChatCompletionRequest,
    ) -> Result<MixlayerChatCompletionResponse, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LlmError::authentication("Invalid API key format"))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let response = self
            .http_client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|source| LlmError::Network { source })?;
        let status = response.status();
        if status.is_success() {
            return response
                .json()
                .await
                .map_err(|e| LlmError::internal(format!("Failed to parse response: {}", e)));
        }

        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".into());
        let message = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                v.pointer("/error/message")
                    .and_then(|m| m.as_str())
                    .map(str::to_owned)
            })
            .unwrap_or(body);
        match status {
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
                Err(LlmError::authentication(message))
            }
            reqwest::StatusCode::BAD_REQUEST | reqwest::StatusCode::PAYLOAD_TOO_LARGE => {
                Err(LlmError::invalid_request(message))
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => Err(LlmError::rate_limit(message, None)),
            _ => Err(LlmError::api_error(status.as_u16(), message)),
        }
    }
}

impl MixlayerChatCompletionResponse {
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
                        crate::tools::ToolCall::new(
                            call.id.clone(),
                            call.function.name.clone(),
                            serde_json::from_str(&call.function.arguments)
                                .unwrap_or(serde_json::Value::Null),
                        )
                    })
                    .collect()
            })
    }
}

#[async_trait]
impl crate::client::LlmClient for MixlayerClient {
    async fn complete(
        &self,
        request: crate::types::CompletionRequest,
    ) -> Result<crate::types::CompletionResponse, LlmError> {
        let mut messages = Vec::new();
        if let Some(system) = request.system {
            messages.push(MixlayerMessage::system(system));
        }
        for message in request.messages {
            let content = message
                .content
                .into_iter()
                .map(|block| match block {
                    crate::types::ContentBlock::Text { text } => Ok(text),
                    crate::types::ContentBlock::Image { .. } => Err(LlmError::invalid_request(
                        "Image content not supported by MixLayer client",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("");
            let provider_message = match message.role {
                crate::types::Role::Assistant if message.tool_call_id.is_some() => {
                    MixlayerMessage {
                        role: MixlayerRole::Assistant,
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(vec![MixlayerToolCall {
                            id: message.tool_call_id.unwrap(),
                            tool_type: "function".into(),
                            function: MixlayerFunctionCall {
                                name: message.tool_name.unwrap_or_default(),
                                arguments: content,
                            },
                            index: None,
                        }]),
                        tool_call_id: None,
                        sources: None,
                    }
                }
                crate::types::Role::Tool => {
                    MixlayerMessage::tool_result(message.tool_call_id.unwrap_or_default(), content)
                }
                role => MixlayerMessage::new(
                    match role {
                        crate::types::Role::User => MixlayerRole::User,
                        crate::types::Role::Assistant => MixlayerRole::Assistant,
                        crate::types::Role::System => MixlayerRole::System,
                        crate::types::Role::Tool => MixlayerRole::Tool,
                    },
                    content,
                ),
            };
            messages.push(provider_message);
        }
        let tools = request.tools.map(|tools| {
            tools
                .iter()
                .map(MixlayerToolFormat::to_provider_tool)
                .collect()
        });
        let tool_choice = request
            .tool_choice
            .map(|choice| MixlayerToolFormat::to_provider_tool_choice(&choice));
        let response_format = request.response_format.map(|format| match format {
            crate::types::ResponseFormat::Text => MixlayerResponseFormat::text(),
            crate::types::ResponseFormat::JsonObject => MixlayerResponseFormat::json_object(),
        });
        let response = self
            .create_chat_completion(MixlayerChatCompletionRequest {
                model: if request.model.is_empty() {
                    self.model.clone()
                } else {
                    request.model
                },
                messages,
                max_completion_tokens: Some(request.max_tokens),
                max_tokens: None,
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: None,
                frequency_penalty: None,
                presence_penalty: None,
                repetition_penalty: None,
                stop: request.stop_sequences,
                seed: None,
                reasoning_effort: None,
                thinking: None,
                response_format,
                tools,
                tool_choice,
                web_search_options: None,
                metadata: None,
                store: None,
            })
            .await?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::internal("No completion choices returned"))?;
        let tool_calls = response.tool_calls().filter(|calls| !calls.is_empty());
        Ok(crate::types::CompletionResponse {
            content: vec![crate::types::ContentBlock::Text {
                text: choice.message.content.clone().unwrap_or_default(),
            }],
            role: match choice.message.role {
                MixlayerRole::System => crate::types::Role::System,
                MixlayerRole::User => crate::types::Role::User,
                MixlayerRole::Assistant => crate::types::Role::Assistant,
                MixlayerRole::Tool => crate::types::Role::Tool,
            },
            usage: crate::types::Usage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
            },
            stop_reason: choice.finish_reason.clone(),
            tool_calls,
        })
    }

    fn provider_name(&self) -> &str {
        crate::providers::MIXLAYER
    }
    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LlmClient;

    #[tokio::test]
    async fn sends_bearer_auth_and_maps_response() {
        let mut server = mockito::Server::new_async().await;
        let mock = server.mock("POST", "/v1/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "model": "qwen/qwen3.5-4b-free",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 32
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"id":"c1","object":"chat.completion","created":1,"model":"qwen/qwen3.5-4b-free","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,"prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0}}}"#)
            .create_async().await;
        let response = MixlayerClient::new("test-key")
            .unwrap()
            .with_base_url(server.url())
            .message_builder()
            .user_message("hello")
            .max_completion_tokens(32)
            .send()
            .await
            .unwrap();
        mock.assert_async().await;
        assert_eq!(response.choices[0].message.content.as_deref(), Some("hi"));
    }

    #[test]
    fn has_expected_provider_and_default_model() {
        let client = MixlayerClient::new("test-key").unwrap();
        assert_eq!(client.provider_name(), "mixlayer");
        assert_eq!(
            LlmClient::model_name(&client),
            crate::models::mixlayer::QWEN_3_5_4B_FREE_ID
        );
    }
}
