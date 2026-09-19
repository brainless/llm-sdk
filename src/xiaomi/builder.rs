use crate::{
    error::LlmError,
    tools::{ProviderToolFormat, Tool, ToolChoice, ToolResult},
    xiaomi::{
        client::XiaomiClient,
        tools::XiaomiToolFormat,
        types::{
            XiaomiAsrLanguage, XiaomiAsrOptions, XiaomiAudioFormat, XiaomiChatCompletionRequest,
            XiaomiChatCompletionResponse, XiaomiInputAudio, XiaomiMessage, XiaomiResponseFormat,
            XiaomiSpeechRecognitionMessage, XiaomiSpeechRecognitionRequest, XiaomiTool,
        },
    },
};

/// Builder for a Xiaomi Chat Completions request.
pub struct XiaomiMessageBuilder<'a> {
    client: &'a XiaomiClient,
    model: Option<String>,
    messages: Vec<XiaomiMessage>,
    max_completion_tokens: Option<u32>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    response_format: Option<XiaomiResponseFormat>,
    tools: Option<Vec<XiaomiTool>>,
    tool_choice: Option<serde_json::Value>,
}

impl<'a> XiaomiMessageBuilder<'a> {
    pub fn new(client: &'a XiaomiClient) -> Self {
        Self {
            client,
            model: None,
            messages: Vec::new(),
            max_completion_tokens: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            response_format: None,
            tools: None,
            tool_choice: None,
        }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn system_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(XiaomiMessage::system(content));
        self
    }

    pub fn user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(XiaomiMessage::user(content));
        self
    }

    pub fn assistant_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(XiaomiMessage::assistant(content));
        self
    }

    pub fn tool_result(mut self, result: ToolResult) -> Self {
        self.messages.push(XiaomiMessage::tool_result(
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

    pub fn stop_sequences(mut self, value: Vec<String>) -> Self {
        self.stop = Some(value);
        self
    }

    pub fn response_format(mut self, value: XiaomiResponseFormat) -> Self {
        self.response_format = Some(value);
        self
    }

    pub fn json_schema_response_format(self, schema: serde_json::Value) -> Self {
        self.response_format(XiaomiResponseFormat::json_schema(schema))
    }

    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools
            .get_or_insert_with(Vec::new)
            .push(XiaomiToolFormat::to_provider_tool(&tool));
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        for tool in tools {
            self = self.tool(tool);
        }
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(XiaomiToolFormat::to_provider_tool_choice(&choice));
        self
    }

    pub async fn send(self) -> Result<XiaomiChatCompletionResponse, LlmError> {
        let request = XiaomiChatCompletionRequest {
            model: self
                .model
                .unwrap_or_else(|| self.client.model_name().to_string()),
            messages: self.messages,
            max_completion_tokens: self.max_completion_tokens,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            stop: self.stop,
            response_format: self.response_format,
            tools: self.tools,
            tool_choice: self.tool_choice,
        };
        self.client.create_chat_completion(request).await
    }
}

/// Builder for a Xiaomi speech recognition request.
pub struct XiaomiSpeechRecognitionBuilder<'a> {
    client: &'a XiaomiClient,
    model: Option<String>,
    input_audio: Option<XiaomiInputAudio>,
    language: Option<XiaomiAsrLanguage>,
}

impl<'a> XiaomiSpeechRecognitionBuilder<'a> {
    pub fn new(client: &'a XiaomiClient) -> Self {
        Self {
            client,
            model: None,
            input_audio: None,
            language: None,
        }
    }

    /// Override the speech recognition model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set a complete provider-native audio input.
    pub fn input_audio(mut self, input_audio: XiaomiInputAudio) -> Self {
        self.input_audio = Some(input_audio);
        self
    }

    /// Set audio as a data URL containing its MIME type and Base64 data.
    pub fn audio_data_url(self, data_url: impl Into<String>) -> Self {
        self.input_audio(XiaomiInputAudio::data_url(data_url))
    }

    /// Set raw Base64 audio data with its explicit format.
    pub fn audio_base64(self, data: impl Into<String>, format: XiaomiAudioFormat) -> Self {
        self.input_audio(XiaomiInputAudio::base64(data, format))
    }

    /// Set the expected audio language. Omit this to use the provider default.
    pub fn language(mut self, language: XiaomiAsrLanguage) -> Self {
        self.language = Some(language);
        self
    }

    pub async fn send(self) -> Result<XiaomiChatCompletionResponse, LlmError> {
        let input_audio = self
            .input_audio
            .ok_or_else(|| LlmError::invalid_request("Audio input is required"))?;
        let request = XiaomiSpeechRecognitionRequest {
            model: self
                .model
                .unwrap_or_else(|| crate::models::xiaomi::MIMO_V2_5_ASR.to_string()),
            messages: vec![XiaomiSpeechRecognitionMessage::user(input_audio)],
            asr_options: self.language.map(|language| XiaomiAsrOptions { language }),
        };
        self.client.create_speech_recognition(request).await
    }
}
