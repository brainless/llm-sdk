use llm_sdk::models::xiaomi::{MIMO_V2_5, MIMO_V2_5_ASR, MIMO_V2_5_PRO};
use llm_sdk::xiaomi::{XiaomiAsrLanguage, XiaomiAudioFormat, XiaomiClient};

// Integration tests require XIAOMI_API_KEY environment variable
// Run with: XIAOMI_API_KEY=... cargo test --test xiaomi_integration -- --ignored
// The ASR test also requires raw WAV Base64 in XIAOMI_ASR_AUDIO_BASE64 (no data-URL prefix).

fn get_api_key() -> Option<String> {
    std::env::var("XIAOMI_API_KEY").ok()
}

fn skip_if_no_api_key() {
    if get_api_key().is_none() {
        panic!("Skipping integration test - XIAOMI_API_KEY not set");
    }
}

#[tokio::test]
#[ignore] // Run manually with API key
async fn test_xiaomi_mimo_v2_5() {
    skip_if_no_api_key();
    let api_key = get_api_key().unwrap();

    let client = XiaomiClient::new(api_key).unwrap();
    let response = client
        .message_builder()
        .model(MIMO_V2_5)
        .max_completion_tokens(100)
        .user_message("Say 'Hello, World!' and nothing else.")
        .send()
        .await;

    assert!(response.is_ok());
    let response = response.unwrap();
    assert!(!response.choices.is_empty());
    assert!(!response.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_default()
        .is_empty());
}

#[tokio::test]
#[ignore] // Run manually with API key
async fn test_xiaomi_mimo_v2_5_pro() {
    skip_if_no_api_key();
    let api_key = get_api_key().unwrap();

    let client = XiaomiClient::new(api_key).unwrap();
    let response = client
        .message_builder()
        .model(MIMO_V2_5_PRO)
        .max_completion_tokens(100)
        .user_message("Say 'Hello, World!' and nothing else.")
        .send()
        .await;

    assert!(response.is_ok());
    let response = response.unwrap();
    assert!(!response.choices.is_empty());
    assert!(!response.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_default()
        .is_empty());
}

#[tokio::test]
#[ignore] // Run manually with an API key and Base64-encoded WAV input
async fn test_xiaomi_mimo_v2_5_asr() {
    skip_if_no_api_key();
    let api_key = get_api_key().unwrap();
    let audio_base64 = std::env::var("XIAOMI_ASR_AUDIO_BASE64")
        .expect("XIAOMI_ASR_AUDIO_BASE64 must contain raw Base64 for a WAV file");

    let client = XiaomiClient::new(api_key).unwrap();
    let response = client
        .speech_recognition_builder()
        .model(MIMO_V2_5_ASR)
        .audio_base64(audio_base64, XiaomiAudioFormat::Wav)
        .language(XiaomiAsrLanguage::Auto)
        .send()
        .await;

    assert!(response.is_ok());
    let response = response.unwrap();
    assert!(!response.choices.is_empty());
    assert!(!response.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or_default()
        .is_empty());
}

#[tokio::test]
#[ignore] // Run manually with API key
async fn test_xiaomi_invalid_api_key() {
    let client = XiaomiClient::new("invalid-key").unwrap();
    let response = client
        .message_builder()
        .model(MIMO_V2_5)
        .max_completion_tokens(100)
        .user_message("Hello")
        .send()
        .await;

    assert!(response.is_err());
}
