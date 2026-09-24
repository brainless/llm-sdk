//! ElevenLabs Scribe speech-to-text API.

mod client;
mod types;

pub use crate::models::elevenlabs::*;
pub use client::ElevenLabsClient;
pub use types::*;
