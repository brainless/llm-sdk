use llm_sdk::openrouter::OpenRouterClient;

#[tokio::test]
async fn test_openrouter_empty_api_key() {
    let result = OpenRouterClient::new("");
    assert!(result.is_err());
    match result.unwrap_err() {
        llm_sdk::error::LlmError::Authentication { .. } => {}
        other => panic!("Expected authentication error, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_openrouter_no_model_complete_fails() {
    use llm_sdk::client::LlmClient;
    use llm_sdk::types::{CompletionRequest, Message, Role, ContentBlock};

    let client = OpenRouterClient::new("fake-key").expect("Failed to create client");

    let request = CompletionRequest {
        model: String::new(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Hello".to_string() }],
            tool_call_id: None,
            tool_name: None,
        }],
        system: None,
        max_tokens: 10,
        temperature: None,
        top_p: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
        response_format: None,
    };

    let result = client.complete(request).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        llm_sdk::error::LlmError::InvalidRequest { .. } => {}
        other => panic!("Expected invalid request error, got: {:?}", other),
    }
}

#[tokio::test]
#[ignore] // Requires OPENROUTER_API_KEY environment variable
async fn test_list_free_programming_models() {
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key).expect("Failed to create client");

    let models = client
        .list_free_programming_models()
        .await
        .expect("Failed to list models");

    assert!(
        !models.is_empty(),
        "Expected at least one free programming model"
    );

    for model in &models {
        assert!(
            model.pricing.is_free(),
            "Model {} is not free: prompt={} completion={}",
            model.id,
            model.pricing.prompt,
            model.pricing.completion
        );
    }

    println!("Free programming models ({}):", models.len());
    for model in &models {
        println!(
            "  {} — {} (context: {:?})",
            model.id, model.name, model.context_length
        );
    }
}

#[tokio::test]
#[ignore] // Requires OPENROUTER_API_KEY environment variable
async fn test_complete_with_free_model() {
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key).expect("Failed to create client");

    let models = client
        .list_free_programming_models()
        .await
        .expect("Failed to list models");

    assert!(!models.is_empty(), "No free programming models available");

    let model_id = models[0].id.clone();
    println!("Testing completion with model: {}", model_id);

    let response = client
        .with_model(&model_id)
        .message_builder()
        .model(&model_id)
        .max_tokens(64)
        .user_message("Say 'Hello, World!' and nothing else.")
        .temperature(0.0)
        .send()
        .await
        .expect("Failed to get completion");

    assert!(!response.choices.is_empty());
    let text = &response.choices[0].message.content;
    assert!(!text.trim().is_empty(), "Response should not be empty");
    println!("Response: {}", text);
}

#[tokio::test]
#[ignore] // Requires OPENROUTER_API_KEY environment variable
async fn test_complete_via_llm_client_trait() {
    use llm_sdk::client::LlmClient;
    use llm_sdk::types::{CompletionRequest, Message, Role, ContentBlock};

    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key).expect("Failed to create client");

    let models = client
        .list_free_programming_models()
        .await
        .expect("Failed to list models");

    assert!(!models.is_empty(), "No free programming models available");

    let client = client.with_model(&models[0].id);

    assert_eq!(client.provider_name(), "openrouter");
    assert_eq!(client.model_name(), models[0].id.as_str());

    let request = CompletionRequest {
        model: models[0].id.clone(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Say 'Hello, World!' and nothing else.".to_string(),
            }],
            tool_call_id: None,
            tool_name: None,
        }],
        system: None,
        max_tokens: 64,
        temperature: Some(0.0),
        top_p: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
        response_format: None,
    };

    let response = client.complete(request).await.expect("Completion failed");

    assert!(!response.content.is_empty());
    println!("LlmClient trait response: {:?}", response.content);
}
