use reqwest::{multipart, StatusCode};

use crate::error::LlmError;

use super::{
    MultichannelTranscript, Transcript, TranscriptionRequest, TranscriptionResult,
    TranscriptionSource, WebhookAcceptance,
};

/// Client for ElevenLabs Scribe batch speech-to-text transcription.
#[derive(Debug)]
pub struct ElevenLabsClient {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl ElevenLabsClient {
    /// Create a client using an ElevenLabs API key.
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
            base_url: "https://api.elevenlabs.io".into(),
            http_client,
        })
    }

    /// Select a regional or custom API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_owned();
        self
    }

    /// Transcribe a file or hosted media URL with Scribe v2.
    pub async fn transcribe(
        &self,
        request: TranscriptionRequest,
    ) -> Result<TranscriptionResult, LlmError> {
        validate(&request)?;
        let mut form = multipart::Form::new().text("model_id", request.model_id);
        form = match request.source {
            TranscriptionSource::File { bytes, filename } => {
                form.part("file", multipart::Part::bytes(bytes).file_name(filename))
            }
            TranscriptionSource::Url(url) => form.text("source_url", url),
        };
        macro_rules! field {
            ($name:literal, $value:expr) => {
                if let Some(value) = $value {
                    form = form.text($name, value.to_string());
                }
            };
        }
        field!("language_code", request.language_code);
        field!("tag_audio_events", request.tag_audio_events);
        field!("num_speakers", request.num_speakers);
        field!(
            "timestamps_granularity",
            request.timestamps_granularity.map(|v| v.as_str())
        );
        field!("diarize", request.diarize);
        field!("diarization_threshold", request.diarization_threshold);
        field!("no_verbatim", request.no_verbatim);
        field!("file_format", request.file_format.map(|v| v.as_str()));
        field!("use_speaker_library", request.use_speaker_library);
        field!("detect_speaker_roles", request.detect_speaker_roles);
        field!("use_multi_channel", request.use_multi_channel);
        field!(
            "multichannel_output_style",
            request.multichannel_output_style.map(|v| v.as_str())
        );
        field!("temperature", request.temperature);
        field!("seed", request.seed);
        field!("webhook", request.webhook);
        field!("webhook_id", request.webhook_id);
        if let Some(metadata) = request.webhook_metadata {
            form = form.text("webhook_metadata", metadata.to_string());
        }
        for term in request.keyterms {
            form = form.text("keyterms", term);
        }

        let mut http_request = self
            .http_client
            .post(format!("{}/v1/speech-to-text", self.base_url))
            .header("xi-api-key", &self.api_key)
            .multipart(form);
        if let Some(enable_logging) = request.enable_logging {
            http_request = http_request.query(&[("enable_logging", enable_logging)]);
        }
        let response = http_request
            .send()
            .await
            .map_err(|source| LlmError::Network { source })?;
        let status = response.status();
        if !status.is_success() {
            // Provider errors may echo signed source URLs or other private input.
            return Err(match status {
                StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                    LlmError::authentication("ElevenLabs rejected the API key")
                }
                StatusCode::BAD_REQUEST
                | StatusCode::UNPROCESSABLE_ENTITY
                | StatusCode::PAYLOAD_TOO_LARGE => {
                    LlmError::invalid_request(format!("ElevenLabs rejected request ({status})"))
                }
                StatusCode::TOO_MANY_REQUESTS => {
                    let retry_after = response
                        .headers()
                        .get("retry-after")
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.parse().ok());
                    LlmError::rate_limit("ElevenLabs rate limit exceeded", retry_after)
                }
                _ => LlmError::api_error(status.as_u16(), "ElevenLabs request failed".into()),
            });
        }
        let value: serde_json::Value = response.json().await.map_err(|source| {
            LlmError::internal(format!("Failed to parse ElevenLabs response: {source}"))
        })?;
        if status == StatusCode::ACCEPTED {
            return serde_json::from_value::<WebhookAcceptance>(value)
                .map(TranscriptionResult::Accepted)
                .map_err(LlmError::from);
        }
        if value.get("transcripts").is_some() {
            serde_json::from_value::<MultichannelTranscript>(value)
                .map(TranscriptionResult::Multichannel)
                .map_err(LlmError::from)
        } else {
            serde_json::from_value::<Transcript>(value)
                .map(TranscriptionResult::Transcript)
                .map_err(LlmError::from)
        }
    }
}

fn validate(request: &TranscriptionRequest) -> Result<(), LlmError> {
    if request.model_id.trim().is_empty() {
        return Err(LlmError::invalid_request("model_id cannot be empty"));
    }
    match &request.source {
        TranscriptionSource::File { bytes, filename }
            if bytes.is_empty() || filename.is_empty() =>
        {
            return Err(LlmError::invalid_request(
                "file bytes and filename are required",
            ));
        }
        TranscriptionSource::Url(url) if url.is_empty() => {
            return Err(LlmError::invalid_request("source_url cannot be empty"));
        }
        _ => {}
    }
    if request.num_speakers.is_some_and(|n| n == 0 || n > 32) {
        return Err(LlmError::invalid_request("num_speakers must be 1..=32"));
    }
    if request.seed.is_some_and(|seed| seed > i32::MAX as u32) {
        return Err(LlmError::invalid_request("seed must be 0..=2147483647"));
    }
    if request
        .diarization_threshold
        .is_some_and(|v| !(0.1..=0.4).contains(&v))
    {
        return Err(LlmError::invalid_request(
            "diarization_threshold must be 0.1..=0.4",
        ));
    }
    if request
        .temperature
        .is_some_and(|v| !(0.0..=2.0).contains(&v))
    {
        return Err(LlmError::invalid_request("temperature must be 0.0..=2.0"));
    }
    if request.webhook_id.is_some() && request.webhook != Some(true) {
        return Err(LlmError::invalid_request(
            "webhook_id requires webhook=true",
        ));
    }
    if request.diarization_threshold.is_some()
        && (request.diarize != Some(true) || request.num_speakers.is_some())
    {
        return Err(LlmError::invalid_request(
            "diarization_threshold requires diarize=true and no num_speakers",
        ));
    }
    if request.detect_speaker_roles == Some(true)
        && (request.diarize != Some(true) || request.use_multi_channel == Some(true))
    {
        return Err(LlmError::invalid_request(
            "detect_speaker_roles requires diarize=true and no multichannel input",
        ));
    }
    if request.use_speaker_library == Some(true) && request.diarize != Some(true) {
        return Err(LlmError::invalid_request(
            "use_speaker_library requires diarize=true",
        ));
    }
    if matches!(
        request.multichannel_output_style,
        Some(super::MultichannelOutputStyle::Combined)
    ) && (request.use_multi_channel != Some(true)
        || matches!(
            request.timestamps_granularity,
            Some(super::TimestampsGranularity::None)
        ))
    {
        return Err(LlmError::invalid_request(
            "combined multichannel output requires use_multi_channel=true and timestamps",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{models, providers};
    use mockito::Matcher;

    const SINGLE: &str = r#"{"language_code":"en","language_probability":0.98,"text":"Hello","words":[{"text":"Hello","type":"word","logprob":-0.1,"start":0,"end":0.5,"speaker_id":"speaker_0","characters":[{"text":"H","start":0,"end":0.1}]}],"entities":[{"text":"Hello","entity_type":"person_name","start_char":0,"end_char":5}]}"#;

    #[tokio::test]
    async fn uploads_file_and_parses_transcript() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/speech-to-text")
            .match_header("xi-api-key", "secret")
            .match_header(
                "content-type",
                Matcher::Regex("^multipart/form-data; boundary=".into()),
            )
            .match_body(Matcher::Regex(
                "name=\"model_id\"\\r\\n\\r\\nscribe_v2".into(),
            ))
            .match_body(Matcher::Regex("filename=\"clip.wav\"".into()))
            .match_body(Matcher::Regex("name=\"keyterms\"\\r\\n\\r\\nAcme".into()))
            .with_status(200)
            .with_body(SINGLE)
            .create_async()
            .await;
        let client = ElevenLabsClient::new("secret")
            .unwrap()
            .with_base_url(server.url());
        let mut request = TranscriptionRequest::file(b"wav".to_vec(), "clip.wav");
        request.keyterms.push("Acme".into());
        let TranscriptionResult::Transcript(transcript) = client.transcribe(request).await.unwrap()
        else {
            panic!("expected transcript")
        };
        assert_eq!(transcript.text, "Hello");
        assert_eq!(
            transcript.words[0].characters.as_ref().unwrap()[0].text,
            "H"
        );
        assert_eq!(transcript.entities.unwrap()[0].entity_type, "person_name");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn sends_url_and_parses_multichannel_transcripts() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/speech-to-text?enable_logging=false")
            .match_body(Matcher::Regex(
                "name=\"model_id\"\\r\\n\\r\\nscribe_v2_medical".into(),
            ))
            .match_body(Matcher::Regex(
                "name=\"source_url\"\\r\\n\\r\\nhttps://example.com/clip.mp3".into(),
            ))
            .match_body(Matcher::Regex(
                "name=\"use_multi_channel\"\\r\\n\\r\\ntrue".into(),
            ))
            .with_status(200)
            .with_body(format!(
                r#"{{"transcripts":[{}],"audio_duration_secs":1.2}}"#,
                SINGLE
            ))
            .create_async()
            .await;
        let client = ElevenLabsClient::new("secret")
            .unwrap()
            .with_base_url(server.url());
        let mut request = TranscriptionRequest::source_url("https://example.com/clip.mp3");
        request.model_id = models::elevenlabs::SCRIBE_V2_MEDICAL_ID.into();
        request.use_multi_channel = Some(true);
        request.enable_logging = Some(false);
        let TranscriptionResult::Multichannel(result) = client.transcribe(request).await.unwrap()
        else {
            panic!("expected multichannel transcript")
        };
        assert_eq!(result.transcripts.len(), 1);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn handles_webhook_acceptance_and_sanitizes_errors() {
        let mut server = mockito::Server::new_async().await;
        let accepted = server
            .mock("POST", "/v1/speech-to-text")
            .match_body(Matcher::Regex("name=\"webhook\"\\r\\n\\r\\ntrue".into()))
            .with_status(202)
            .with_body(r#"{"message":"accepted","request_id":"r1"}"#)
            .create_async()
            .await;
        let client = ElevenLabsClient::new("secret")
            .unwrap()
            .with_base_url(server.url());
        let mut request = TranscriptionRequest::source_url("https://example.com/clip.mp3");
        request.webhook = Some(true);
        let TranscriptionResult::Accepted(result) = client.transcribe(request).await.unwrap()
        else {
            panic!("expected acceptance")
        };
        assert_eq!(result.request_id, "r1");
        accepted.assert_async().await;

        let bad = server
            .mock("POST", "/v1/speech-to-text")
            .with_status(422)
            .with_body("private signed URL and transcript")
            .create_async()
            .await;
        let err = client
            .transcribe(TranscriptionRequest::source_url(
                "https://example.com/clip.mp3",
            ))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::InvalidRequest { .. }));
        assert!(!err.to_string().contains("private"));
        bad.assert_async().await;
    }

    #[test]
    fn validates_request_and_exports_model() {
        assert_eq!(providers::ELEVENLABS, "elevenlabs");
        assert_eq!(models::elevenlabs::SCRIBE_V2_ID, "scribe_v2");
        assert_eq!(models::elevenlabs::SCRIBE_V2_NAME, "Scribe v2");
        assert_eq!(
            models::elevenlabs::SCRIBE_V2_MEDICAL_ID,
            "scribe_v2_medical"
        );
        assert_eq!(
            models::elevenlabs::SCRIBE_V2_MEDICAL_NAME,
            "Scribe v2 Medical"
        );
        let mut request = TranscriptionRequest::file(Vec::new(), "a.wav");
        assert!(matches!(
            validate(&request),
            Err(LlmError::InvalidRequest { .. })
        ));
        request.source = TranscriptionSource::Url("https://example.com/a.wav".into());
        request.num_speakers = Some(33);
        assert!(matches!(
            validate(&request),
            Err(LlmError::InvalidRequest { .. })
        ));
    }
}
