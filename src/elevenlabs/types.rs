use serde::{Deserialize, Serialize};

/// Audio input for a transcription request.
#[derive(Debug, Clone)]
pub enum TranscriptionSource {
    /// File bytes and a filename sent as multipart form data.
    File { bytes: Vec<u8>, filename: String },
    /// Publicly accessible media URL.
    Url(String),
}

/// Timestamp granularity requested from Scribe.
#[derive(Debug, Clone, Copy)]
pub enum TimestampsGranularity {
    Word,
    Character,
    None,
}

impl TimestampsGranularity {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Word => "word",
            Self::Character => "character",
            Self::None => "none",
        }
    }
}

/// How multichannel transcripts are returned.
#[derive(Debug, Clone, Copy)]
pub enum MultichannelOutputStyle {
    Separate,
    Combined,
}

/// Encoding of uploaded audio; PCM requires 16 kHz mono signed 16-bit little-endian samples.
#[derive(Debug, Clone, Copy)]
pub enum InputFileFormat {
    PcmS16le16,
    Other,
}

impl InputFileFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PcmS16le16 => "pcm_s16le_16",
            Self::Other => "other",
        }
    }
}

impl MultichannelOutputStyle {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::Combined => "combined",
        }
    }
}

/// Provider-native Scribe transcription request.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    pub model_id: String,
    pub source: TranscriptionSource,
    pub language_code: Option<String>,
    pub tag_audio_events: Option<bool>,
    pub num_speakers: Option<u8>,
    pub timestamps_granularity: Option<TimestampsGranularity>,
    pub diarize: Option<bool>,
    pub diarization_threshold: Option<f64>,
    pub no_verbatim: Option<bool>,
    pub file_format: Option<InputFileFormat>,
    pub use_speaker_library: Option<bool>,
    pub detect_speaker_roles: Option<bool>,
    pub keyterms: Vec<String>,
    pub use_multi_channel: Option<bool>,
    pub multichannel_output_style: Option<MultichannelOutputStyle>,
    pub temperature: Option<f64>,
    pub seed: Option<u32>,
    pub webhook: Option<bool>,
    pub webhook_id: Option<String>,
    pub webhook_metadata: Option<serde_json::Value>,
    pub enable_logging: Option<bool>,
}

impl TranscriptionRequest {
    /// Start a Scribe v2 request with uploaded file bytes.
    pub fn file(bytes: impl Into<Vec<u8>>, filename: impl Into<String>) -> Self {
        Self::new(TranscriptionSource::File {
            bytes: bytes.into(),
            filename: filename.into(),
        })
    }

    /// Start a Scribe v2 request with a hosted media URL.
    pub fn source_url(url: impl Into<String>) -> Self {
        Self::new(TranscriptionSource::Url(url.into()))
    }

    fn new(source: TranscriptionSource) -> Self {
        Self {
            model_id: crate::models::elevenlabs::SCRIBE_V2_ID.into(),
            source,
            language_code: None,
            tag_audio_events: None,
            num_speakers: None,
            timestamps_granularity: None,
            diarize: None,
            diarization_threshold: None,
            no_verbatim: None,
            file_format: None,
            use_speaker_library: None,
            detect_speaker_roles: None,
            keyterms: Vec::new(),
            use_multi_channel: None,
            multichannel_output_style: None,
            temperature: None,
            seed: None,
            webhook: None,
            webhook_id: None,
            webhook_metadata: None,
            enable_logging: None,
        }
    }
}

/// A synchronous transcript or an accepted asynchronous webhook request.
#[derive(Debug, Clone)]
pub enum TranscriptionResult {
    Transcript(Transcript),
    Multichannel(MultichannelTranscript),
    Accepted(WebhookAcceptance),
}

/// Transcript for one channel or a combined multichannel result.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Transcript {
    pub language_code: String,
    pub language_probability: f64,
    pub text: String,
    pub words: Vec<TranscriptWord>,
    #[serde(default)]
    pub channel_index: Option<u8>,
    #[serde(default)]
    pub additional_formats: Option<Vec<AdditionalFormat>>,
    #[serde(default)]
    pub transcription_id: Option<String>,
    #[serde(default)]
    pub entities: Option<Vec<DetectedEntity>>,
    #[serde(default)]
    pub audio_duration_secs: Option<f64>,
}

/// Separate transcripts for each audio channel.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultichannelTranscript {
    pub transcripts: Vec<Transcript>,
    #[serde(default)]
    pub transcription_id: Option<String>,
    #[serde(default)]
    pub audio_duration_secs: Option<f64>,
}

/// Acceptance response for an asynchronous webhook request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WebhookAcceptance {
    pub message: String,
    pub request_id: String,
    #[serde(default)]
    pub transcription_id: Option<String>,
}

/// Transcribed word, spacing, or audio event.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptWord {
    pub text: String,
    #[serde(rename = "type")]
    pub kind: TranscriptWordType,
    pub logprob: f64,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub speaker_id: Option<String>,
    #[serde(default)]
    pub characters: Option<Vec<TranscriptCharacter>>,
    #[serde(default)]
    pub channel_index: Option<u8>,
}

/// Kind of transcription segment.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptWordType {
    Word,
    Spacing,
    AudioEvent,
}

/// Character-level timing information.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TranscriptCharacter {
    pub text: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

/// An exported transcript format.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AdditionalFormat {
    pub requested_format: String,
    pub file_extension: String,
    pub content_type: String,
    pub is_base64_encoded: bool,
    pub content: String,
}

/// Entity detected in a transcript.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DetectedEntity {
    pub text: String,
    pub entity_type: String,
    pub start_char: usize,
    pub end_char: usize,
}
