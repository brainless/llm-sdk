use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

use crate::{
    error::LlmError,
    tools::ProviderToolFormat,
    xiaomi::{
        tools::XiaomiToolFormat,
        types::{
            XiaomiChatCompletionRequest, XiaomiChatCompletionResponse, XiaomiFunctionCall,
            XiaomiMessage, XiaomiResponseFormat, XiaomiRole, XiaomiSpeechRecognitionRequest,
            XiaomiToolCall,
        },
    },
};

/// Client for Xiaomi MiMo's OpenAI-compatible Chat Completions API.
///
/// Default base URL: `https://token-plan-sgp.xiaomimimo.com/v1`
#[derive(Debug)]
pub struct XiaomiClient {
    api_key: String,
    model: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl XiaomiClient {
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
            model: crate::models::xiaomi::MIMO_V2_5_ID.to_string(),
            base_url: "https://token-plan-sgp.xiaomimimo.com/v1".to_string(),
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

    pub fn message_builder(&self) -> crate::xiaomi::builder::XiaomiMessageBuilder<'_> {
        crate::xiaomi::builder::XiaomiMessageBuilder::new(self)
    }

    /// Start building a MiMo speech recognition request.
    pub fn speech_recognition_builder(
        &self,
    ) -> crate::xiaomi::builder::XiaomiSpeechRecognitionBuilder<'_> {
        crate::xiaomi::builder::XiaomiSpeechRecognitionBuilder::new(self)
    }

    pub async fn create_chat_completion(
        &self,
        request: XiaomiChatCompletionRequest,
    ) -> Result<XiaomiChatCompletionResponse, LlmError> {
        self.send_chat_completion(&request).await
    }

    /// Send a provider-native speech recognition request.
    pub async fn create_speech_recognition(
        &self,
        request: XiaomiSpeechRecognitionRequest,
    ) -> Result<XiaomiChatCompletionResponse, LlmError> {
        self.send_chat_completion(&request).await
    }

    async fn send_chat_completion<T>(
        &self,
        request: &T,
    ) -> Result<XiaomiChatCompletionResponse, LlmError>
    where
        T: serde::Serialize + ?Sized,
    {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key))
                .map_err(|_| LlmError::authentication("Invalid API key format"))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let response = self
            .http_client
            .post(format!("{}/chat/completions", self.base_url))
            .headers(headers)
            .json(request)
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

impl XiaomiChatCompletionResponse {
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
impl crate::client::LlmClient for XiaomiClient {
    async fn complete(
        &self,
        request: crate::types::CompletionRequest,
    ) -> Result<crate::types::CompletionResponse, LlmError> {
        let mut messages = Vec::new();
        if let Some(system) = request.system {
            messages.push(XiaomiMessage::system(system));
        }
        for message in request.messages {
            let content = message
                .content
                .into_iter()
                .map(|block| match block {
                    crate::types::ContentBlock::Text { text } => Ok(text),
                    crate::types::ContentBlock::Image { .. } => Err(LlmError::invalid_request(
                        "Image content not supported by Xiaomi client",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join("");
            let provider_message = match message.role {
                crate::types::Role::Assistant if message.tool_call_id.is_some() => XiaomiMessage {
                    role: XiaomiRole::Assistant,
                    content: None,
                    tool_calls: Some(vec![XiaomiToolCall {
                        id: message.tool_call_id.unwrap(),
                        tool_type: "function".into(),
                        function: XiaomiFunctionCall {
                            name: message.tool_name.unwrap_or_default(),
                            arguments: content,
                        },
                    }]),
                    tool_call_id: None,
                },
                crate::types::Role::Tool => {
                    XiaomiMessage::tool_result(message.tool_call_id.unwrap_or_default(), content)
                }
                role => XiaomiMessage::new(
                    match role {
                        crate::types::Role::User => XiaomiRole::User,
                        crate::types::Role::Assistant => XiaomiRole::Assistant,
                        crate::types::Role::System => XiaomiRole::System,
                        crate::types::Role::Tool => XiaomiRole::Tool,
                    },
                    content,
                ),
            };
            messages.push(provider_message);
        }
        let tools = request.tools.map(|tools| {
            tools
                .iter()
                .map(XiaomiToolFormat::to_provider_tool)
                .collect()
        });
        let tool_choice = request
            .tool_choice
            .map(|choice| XiaomiToolFormat::to_provider_tool_choice(&choice));
        let response_format = request.response_format.map(|format| match format {
            crate::types::ResponseFormat::Text => XiaomiResponseFormat::text(),
            crate::types::ResponseFormat::JsonObject => XiaomiResponseFormat::json_object(),
        });
        let response = self
            .create_chat_completion(XiaomiChatCompletionRequest {
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
                stop: request.stop_sequences,
                response_format,
                tools,
                tool_choice,
                thinking: None,
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
                XiaomiRole::System => crate::types::Role::System,
                XiaomiRole::User => crate::types::Role::User,
                XiaomiRole::Assistant => crate::types::Role::Assistant,
                XiaomiRole::Tool => crate::types::Role::Tool,
            },
            usage: crate::types::Usage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
                reasoning_tokens: None,
            },
            stop_reason: choice.finish_reason.clone(),
            tool_calls,
        })
    }

    fn provider_name(&self) -> &str {
        crate::providers::XIAOMI
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::LlmClient;
    use crate::xiaomi::{XiaomiAsrLanguage, XiaomiAudioFormat};

    #[tokio::test]
    async fn sends_bearer_auth_and_maps_response() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "model": "mimo-v2.5",
                "messages": [{"role": "user", "content": "hello"}],
                "max_completion_tokens": 32
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"id":"c1","object":"chat.completion","created":1,"model":"mimo-v2.5","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}"#)
            .create_async()
            .await;
        let response = XiaomiClient::new("test-key")
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

    #[tokio::test]
    async fn sends_data_url_asr_request_with_default_model_and_language() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .match_header("content-type", "application/json")
            .match_body(mockito::Matcher::Json(serde_json::json!({
                "model": "mimo-v2.5-asr",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": "data:audio/wav;base64,UklGRg=="
                        }
                    }]
                }],
                "asr_options": {"language": "en"}
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"id":"asr1","object":"chat.completion","created":1,"model":"mimo-v2.5-asr","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}"#)
            .create_async()
            .await;

        let response = XiaomiClient::new("test-key")
            .unwrap()
            .with_base_url(server.url())
            .speech_recognition_builder()
            .audio_data_url("data:audio/wav;base64,UklGRg==")
            .language(XiaomiAsrLanguage::En)
            .send()
            .await
            .unwrap();

        mock.assert_async().await;
        assert_eq!(response.model, crate::models::xiaomi::MIMO_V2_5_ASR);
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello")
        );
    }

    #[tokio::test]
    async fn sends_raw_base64_asr_request_without_options() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .match_body(mockito::Matcher::Json(serde_json::json!({
                "model": "custom-asr",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": "UklGRg==",
                            "format": "wav"
                        }
                    }]
                }]
            })))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"id":"asr2","object":"chat.completion","created":1,"model":"custom-asr","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}"#)
            .create_async()
            .await;

        XiaomiClient::new("test-key")
            .unwrap()
            .with_base_url(server.url())
            .speech_recognition_builder()
            .model("custom-asr")
            .audio_base64("UklGRg==", XiaomiAudioFormat::Wav)
            .send()
            .await
            .unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn speech_recognition_requires_audio() {
        let error = XiaomiClient::new("test-key")
            .unwrap()
            .speech_recognition_builder()
            .send()
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            LlmError::InvalidRequest { message } if message == "Audio input is required"
        ));
    }

    #[test]
    fn has_expected_provider_and_default_model() {
        let client = XiaomiClient::new("test-key").unwrap();
        assert_eq!(client.provider_name(), "xiaomi");
        assert_eq!(
            LlmClient::model_name(&client),
            crate::models::xiaomi::MIMO_V2_5_ID
        );
    }
}
