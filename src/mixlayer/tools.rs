use crate::{
    mixlayer::types::{MixlayerFunction, MixlayerTool},
    tools::{ProviderToolFormat, Tool, ToolChoice},
};
use serde_json::{json, Value};

pub struct MixlayerToolFormat;

impl ProviderToolFormat for MixlayerToolFormat {
    type ProviderTool = MixlayerTool;

    fn to_provider_tool(tool: &Tool) -> Self::ProviderTool {
        MixlayerTool {
            tool_type: "function".into(),
            function: MixlayerFunction {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters().clone(),
                strict: None,
            },
        }
    }

    fn to_provider_tool_choice(choice: &ToolChoice) -> Value {
        match choice {
            ToolChoice::Auto => json!("auto"),
            ToolChoice::Required => json!("required"),
            ToolChoice::None => json!("none"),
            ToolChoice::Specific { name } => json!({
                "type": "function",
                "function": { "name": name }
            }),
        }
    }
}
