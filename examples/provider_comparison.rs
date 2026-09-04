//! Example: Comparing different providers for the same models
//!
//! Run with:
//! XAI_API_KEY="..." CEREBRAS_API_KEY="..." cargo run --example provider_comparison

use nocodo_llm_sdk::{
    cerebras::{CerebrasChatCompletionRequest, CerebrasClient, CerebrasMessage, GPT_OSS_120B},
    grok::{
        types::{GrokChatCompletionRequest, GrokMessage, GrokRole},
        xai::XaiGrokClient,
        zen::ZenGrokClient,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Provider Comparison Demo ===\n");

    // Test Grok via different providers
    test_grok_providers().await?;

    println!("\n{}\n", "=".repeat(60));

    // Test Cerebras independently from the model-family comparisons above.
    test_cerebras().await?;

    Ok(())
}

async fn test_grok_providers() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Grok Model via Different Providers ---\n");

    let prompt = "What is 2+2? Answer in one word.";

    // 1. Zen Grok (free)
    println!("1. Zen Provider (FREE):");
    let zen_client = ZenGrokClient::new()?;
    let request = GrokChatCompletionRequest {
        model: "grok-code".to_string(),
        messages: vec![GrokMessage {
            role: GrokRole::User,
            content: prompt.to_string(),
            tool_calls: None,
            tool_call_id: None,
        }],
        max_tokens: Some(50),
        temperature: None,
        top_p: None,
        stop: None,
        tools: None,
        tool_choice: None,
        stream: None,
        response_format: None,
    };

    match zen_client.create_chat_completion(request).await {
        Ok(response) => {
            println!("   Model: {}", response.model);
            println!(
                "   Response: {}",
                response.choices[0].message.content.trim()
            );
        }
        Err(e) => println!("   Error: {}", e),
    }

    // 2. xAI Grok (paid, requires API key)
    println!("\n2. xAI Provider (PAID):");
    if let Ok(api_key) = std::env::var("XAI_API_KEY") {
        let xai_client = XaiGrokClient::new(api_key)?;
        let request = GrokChatCompletionRequest {
            model: "grok-code-fast-1".to_string(),
            messages: vec![GrokMessage {
                role: GrokRole::User,
                content: prompt.to_string(),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: Some(50),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        match xai_client.create_chat_completion(request).await {
            Ok(response) => {
                println!("   Model: {}", response.model);
                println!(
                    "   Response: {}",
                    response.choices[0].message.content.trim()
                );
            }
            Err(e) => println!("   Error: {}", e),
        }
    } else {
        println!("   Skipped (XAI_API_KEY not set)");
    }

    Ok(())
}

async fn test_cerebras() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Cerebras Provider ---\n");

    let prompt = "What is 2+2? Answer in one word.";

    println!("Cerebras Provider:");
    if let Ok(api_key) = std::env::var("CEREBRAS_API_KEY") {
        let cerebras_client = CerebrasClient::new(api_key)?;
        let request = CerebrasChatCompletionRequest {
            model: GPT_OSS_120B.to_string(),
            messages: vec![CerebrasMessage::user(prompt)],
            max_completion_tokens: Some(50),
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            seed: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        match cerebras_client.create_chat_completion(request).await {
            Ok(response) => {
                println!("   Model: {}", response.model);
                println!(
                    "   Response: {}",
                    response.choices[0].message.get_text().trim()
                );
            }
            Err(e) => println!("   Error: {}", e),
        }
    } else {
        println!("   Skipped (CEREBRAS_API_KEY not set)");
    }

    Ok(())
}
