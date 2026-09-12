//! Xiaomi MiMo's OpenAI-compatible Chat Completions API.
//!
//! Default base URL: `https://token-plan-sgp.xiaomimimo.com/v1`

pub mod builder;
pub mod client;
pub mod tools;
pub mod types;

pub use builder::XiaomiMessageBuilder;
pub use client::XiaomiClient;
pub use tools::XiaomiToolFormat;
pub use types::*;

pub use crate::models::xiaomi::*;
