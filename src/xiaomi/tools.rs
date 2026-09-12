use crate::{
    tools::{ProviderToolFormat, Tool, ToolChoice},
    xiaomi::types::{XiaomiFunction, XiaomiTool},
};
use serde_json::{json, Value};

pub struct XiaomiToolFormat;

impl ProviderToolFormat for XiaomiToolFormat {
    type ProviderTool = XiaomiTool;

    fn to_provider_tool(tool: &Tool) -> Self::ProviderTool {
        XiaomiTool {
            tool_type: "function".into(),
            function: XiaomiFunction {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                parameters: tool.parameters().clone(),
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
