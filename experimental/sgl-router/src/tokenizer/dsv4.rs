// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DeepSeek-V4 prompt encoder for cache-aware routing.
//!
//! DeepSeek-V4 ships no Jinja chat template; the engine builds the prompt in
//! code (`python/sglang/srt/entrypoints/openai/encoding_dsv4.py`, selected for
//! the `DeepseekV4` architecture). So to make the router's query tokens match
//! the engine's cached blocks, this reproduces that encoder's output for the
//! routing-relevant subset.
//!
//! # Scope
//!
//! Text content, both chat (non-thinking) and thinking mode — the engine picks
//! per request from `chat_template_kwargs.thinking`, falling back to its
//! `SGLANG_DEFAULT_THINKING`; [`resolve_render_opts`] mirrors that so the router
//! matches whichever mode the engine runs. Chat mode emits a user turn as
//! `BOS <｜User｜> content <｜Assistant｜> </think>`; thinking mode opens the
//! assistant with `<think>` and renders a kept prior assistant turn's
//! `reasoning_content` as `<think>…</think>` (subject to the drop-thinking rules —
//! prior reasoning is dropped without tools, kept with). Preprocessing mirrors
//! `serving_chat.py`/`encode_messages`: an empty system message is inserted when
//! the first message isn't a system message, and `merge_tool_messages` folds
//! `tool` messages into the following user turn (as `<tool_result>` blocks) and
//! coalesces consecutive user turns. Tools, tool calls, and tool results all
//! render the engine's way — the request's `tools` after the system content (see
//! [`render_tools`]), an assistant turn's `tool_calls` as a `DSML` block (see
//! [`render_tool_calls`]), and multiple results ordered by their originating
//! call (see [`sort_tool_results_by_call_order`]). Byte-exactness here is what
//! lets a tool-carrying request's block hashes match the engine's cached blocks
//! instead of diverging from the first block (tools) or the first assistant turn
//! (thinking) and routing by min-load. Two more engine behaviors are mirrored
//! so the ids are engine-equivalent on nearly all dsv4 chat traffic and can be
//! forwarded as `input_ids`: the `task` quick instructions
//! ([`attach_task`], task special tokens in [`render_one`]) and the
//! trailing-assistant surgery behind `continue_final_message`
//! ([`handle_trailing_assistant`], prefix re-appended by the encoder caller,
//! mirroring `_append_assistant_prefix_to_prompt_ids`). What is NOT mirrored —
//! non-text content parts, message-level `wo_eos` / `mask` / client-sent
//! `content_blocks`, and the top-level `reasoning` object — stays withheld
//! from forwarding by `input_ids_safe_to_forward_dsv4`; the render here
//! ignores those keys, so such traffic only degrades routing, never
//! correctness.
//!
//! Tokenization does not auto-prepend special tokens (the `dynamo_tokenizers`
//! HF wrapper hardcodes `add_special_tokens = false`;
//! [`super::adapter::encode`] adds none of its own), so the literal marker text
//! below is what maps to the special token ids. Pinned byte-exact against the
//! live engine's `/tokenize` (DeepSeek-V4-Flash, snapshot `6976c7ff`):
//! `[{user:"ABCD"}]` → `[0, 128803, 51453, 128804, 128822]`.

use serde::Serialize;

/// Beginning-of-sequence marker (token id 0).
const BOS: &str = "<｜begin▁of▁sentence｜>";
/// End-of-sequence marker, closing each prior assistant turn (token id 1).
const EOS: &str = "<｜end▁of▁sentence｜>";
/// User-turn marker (token id 128803).
const USER: &str = "<｜User｜>";
/// Assistant-turn marker, opening the generation prompt (token id 128804).
const ASSISTANT: &str = "<｜Assistant｜>";
/// Thinking-start marker (`encoding_dsv4.thinking_start_token`); a thinking-mode
/// generation prompt / historical assistant turn opens with it.
const THINK_START: &str = "<think>";
/// Thinking-end marker; the chat-mode generation prompt ends with it (128822).
const THINK_END: &str = "</think>";
/// DSML block token wrapping tool-call / tools markup (`encoding_dsv4.dsml_token`).
const DSML: &str = "｜DSML｜";

/// The `encoding_dsv4.DS_TASK_SP_TOKENS` map — special tokens appended after
/// a task-carrying message. Valid task names are exactly these keys; anything
/// else is an engine-side assertion failure (`VALID_TASKS`).
fn task_sp_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some("<｜action｜>"),
        "query" => Some("<｜query｜>"),
        "authority" => Some("<｜authority｜>"),
        "domain" => Some("<｜domain｜>"),
        "title" => Some("<｜title｜>"),
        "read_url" => Some("<｜read_url｜>"),
        _ => None,
    }
}

/// Request-level fields beyond `messages`/`tools` that steer the engine's
/// dsv4 encoding (`serving_chat` dsv4 branch) and which the router must
/// mirror to remain engine-equivalent. Ignored by the Jinja chat encoder.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestParts<'a> {
    /// The request-level `task` (encoding_dsv4's quick-instruction tasks):
    /// attached to the last user/developer message and rendered as a task
    /// special token. `None` for ordinary traffic. (A non-string `task`
    /// fails engine-side pydantic validation, so the wiring treats it as
    /// absent — the engine rejects that request visibly either way.)
    pub task: Option<&'a str>,
    /// OpenAI `continue_final_message` (default false). With a trailing
    /// assistant message the engine extracts it and appends its text AFTER
    /// the generation prompt, so generation continues the client's sentence.
    /// Coerce request values with [`openai_bool`], not a bare `as_bool`.
    pub continue_final_message: bool,
}

/// Parse a JSON value as a coerced OpenAI boolean. Lives in
/// [`crate::tokenizer::openai_bool`]; re-exported here so existing dsv4
/// encoder call sites keep reading naturally.
pub use crate::tokenizer::openai_bool;

/// Errors from engine-mirroring pre-processing. Each corresponds to a case
/// the ENGINE errors on too (a `ValueError`/`AssertionError` in
/// `serving_chat`/`encoding_dsv4`), so the caller treating it as "not
/// engine-equivalent" reproduces the engine's own outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderErr {
    /// `task` present but no user/developer message —
    /// `attach_task_to_last_user_message`'s `ValueError`.
    TaskWithoutUser,
    /// A task name outside `encoding_dsv4.VALID_TASKS` (on the request field
    /// or any message) — the render-time `assert`.
    InvalidTask,
}

impl std::fmt::Display for RenderErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderErr::TaskWithoutUser => write!(f, "`task` requires a user/developer message"),
            RenderErr::InvalidTask => write!(f, "task outside encoding_dsv4.VALID_TASKS"),
        }
    }
}

impl std::error::Error for RenderErr {}

/// Mirror `serving_chat._handle_last_assistant_message` as the engine's dsv4
/// branch applies it. The engine flattens content BEFORE the surgery (string
/// as-is, `null`/absent → `""`, text-parts arrays → their text join — see
/// [`content_to_string`]), so when the trailing message is an assistant turn
/// the surgery sees a string in every one of those shapes:
///   * `continue_final_message = true`: the message is REMOVED and its
///     flattened content handed back as the prefix (the caller encodes it and
///     appends after the generation prompt);
///   * `continue_final_message = false`: the message is REPLACED wholesale
///     with `{"role": "user", "content": flattened}` — dropping every other
///     key (`task`, `tool_calls`, …), not just flipping `role`.
///
/// Content that cannot be flattened (nothing else the caller routes here) is
/// left untouched in both modes, as is any other trailing role.
///
/// Returns the extracted prefix (possibly empty — the engine treats `""` as
/// no prefix via its `if assistant_prefix` check).
fn handle_trailing_assistant(
    messages: &mut Vec<serde_json::Value>,
    continue_final_message: bool,
) -> Option<String> {
    let trailing_role = messages
        .last()
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str());
    if trailing_role != Some("assistant") {
        return None;
    }
    let flattened = match messages.last().and_then(|m| m.get("content")) {
        Some(serde_json::Value::String(s)) => s.clone(),
        // Flattened to "" by the engine before the surgery.
        Some(serde_json::Value::Null) | None => String::new(),
        Some(serde_json::Value::Array(_)) => {
            content_to_string(messages.last().and_then(|m| m.get("content")))
        }
        // Unflattenable content (non-string scalar) — left untouched.
        Some(_) => return None,
    };
    if continue_final_message {
        messages.pop();
        Some(flattened)
    } else {
        *messages.last_mut().unwrap() = serde_json::json!({"role": "user", "content": flattened});
        None
    }
}

/// Mirror `encoding_dsv4.find_last_user_index` + `attach_task_to_last_user_message`:
/// set `task` on the most recent user/developer message. Errs when none
/// exists (the engine's `ValueError`) and when the task name is invalid (the
/// render-time `VALID_TASKS` assert — surfaced here so failure modes are
/// uniform).
fn attach_task(messages: &mut Vec<serde_json::Value>, task: &str) -> Result<(), RenderErr> {
    if task_sp_token(task).is_none() {
        return Err(RenderErr::InvalidTask);
    }
    let idx = messages
        .iter()
        .rposition(|m| {
            matches!(
                m.get("role").and_then(|r| r.as_str()),
                Some("user") | Some("developer")
            )
        })
        .ok_or(RenderErr::TaskWithoutUser)?;
    messages[idx]["task"] = serde_json::json!(task);
    Ok(())
}

/// Render a whole request the way the engine's dsv4 branch does
/// (`serving_chat`): trailing-assistant surgery, then `task` attachment,
/// then [`render_messages`]. Returns the rendered prompt and, when
/// `continue_final_message` extracted a trailing assistant turn, its content
/// — the caller encodes that text and appends the ids after the prompt ids
/// (`_append_assistant_prefix_to_prompt_ids`).
pub fn render_request(
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    opts: RenderOpts,
    parts: RequestParts<'_>,
) -> Result<(String, Option<String>), RenderErr> {
    let mut raw = messages
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .to_vec();
    for m in &mut raw {
        // Protocol message validation normalizes roles to lowercase.
        if let Some(role) = m.get("role").and_then(|r| r.as_str()).map(str::to_owned) {
            m["role"] = serde_json::json!(role.to_ascii_lowercase());
        }
        // Client-sent message-level `task` never reaches encoding_dsv4 (the
        // message model has no such declared field), so it is stripped exactly
        // where the engine strips it — the only task that renders is the
        // request-level one attached below.
        if let serde_json::Value::Object(o) = m {
            o.remove("task");
        }
    }
    let prefix = handle_trailing_assistant(&mut raw, parts.continue_final_message);
    if let Some(task) = parts.task {
        attach_task(&mut raw, task)?;
    }
    // Validate the remaining (attach-set) task keys — the engine's
    // render-time `VALID_TASKS` assert.
    for m in &raw {
        if let Some(t) = m.get("task").and_then(|t| t.as_str()) {
            if task_sp_token(t).is_none() {
                return Err(RenderErr::InvalidTask);
            }
        }
    }
    Ok((
        render_messages(&serde_json::Value::Array(raw), tools, opts),
        prefix,
    ))
}

/// Reasoning-effort level, mirroring `encoding_dsv4`'s `reasoning_effort`
/// (`"max"` / `"high"` / `None`). Only `Max` alters the prompt — it prepends the
/// [`REASONING_EFFORT_MAX`] preamble at the very front in thinking mode. `High` is
/// kept distinct (not collapsed into `None`) intentionally: it is a lossless 1:1
/// mirror of the engine's own tri-state, so a future engine build that gives
/// `high` its own rendering only touches [`render_one`], not this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    High,
    Max,
}

/// How to render, mirroring the engine's per-request `thinking_mode` +
/// `reasoning_effort`. The engine derives these from the request's
/// `chat_template_kwargs.thinking` / `reasoning_effort` (falling back to its
/// `SGLANG_DEFAULT_THINKING` / `SGLANG_DSV4_REASONING_EFFORT` defaults); the
/// router mirrors that resolution in [`resolve_render_opts`] so its routing
/// tokens match the engine's cached blocks even when the engine runs a
/// non-default (e.g. thinking-on) mode. `RenderOpts::chat()` reproduces the
/// chat-mode encoder output byte-for-byte.
#[derive(Clone, Copy, Debug)]
pub struct RenderOpts {
    /// `true` = thinking mode (engine `thinking_mode == "thinking"`).
    pub thinking: bool,
    pub reasoning_effort: ReasoningEffort,
}

impl RenderOpts {
    /// Chat / non-thinking mode with no effort preamble — the engine default,
    /// byte-identical to the chat-mode encoder output.
    pub const fn chat() -> Self {
        RenderOpts {
            thinking: false,
            reasoning_effort: ReasoningEffort::None,
        }
    }
}

/// Resolve the render options for a request, mirroring the deployed engine's
/// dsv4 normalization (`protocol.normalize_reasoning_inputs` +
/// `serving_chat`'s dsv4 branch):
///
/// Thinking: `chat_template_kwargs.thinking` truthiness decides when the key
/// is PRESENT (an explicit `null` counts as present, per `setdefault`
/// semantics). Otherwise top-level `reasoning_effort` defaults it — the
/// protocol validator defaults thinking to `effort != "none"` (a numeric
/// token-budget effort is always `!= "none"`). Otherwise the engine's
/// `SGLANG_DEFAULT_THINKING` env, here the router's env-matching default.
///
/// Reasoning effort: `serving_chat._convert_to_internal_request` pops
/// `chat_template_kwargs.reasoning_effort` ONTO the top-level field BEFORE
/// the dsv4 branch reads it — so the effective precedence is
/// `chat_template_kwargs.reasoning_effort` (present non-null, wins even over
/// a set top-level) > top-level `reasoning_effort` > the engine's
/// `SGLANG_DSV4_REASONING_EFFORT` env, env consulted ONLY when both are
/// absent/null. Any other present value collapses to `None` WITHOUT
/// consulting the env default. Only `max`/`high` alter the prompt. (Note the
/// asymmetry with THINKING: the protocol validator's thinking-setdefault runs
/// at parse time, before the ctk pop, so ctk effort never defaults thinking
/// — only a top-level effort does.)
///
/// The router is a separate process from the engine, so it cannot observe the
/// engine's `SGLANG_DEFAULT_THINKING` / `SGLANG_DSV4_REASONING_EFFORT` env
/// defaults a request doesn't carry; the router's
/// `SGLANG_ROUTER_DSV4_DEFAULT_THINKING` / `SGLANG_ROUTER_DSV4_REASONING_EFFORT`
/// (read once) MUST be set to match them, or routing degrades and plain
/// requests can be forwarded as wrong-mode ids (the predicate can't detect
/// such a mismatch — see `input_ids_safe_to_forward_dsv4`'s deploy note). The
/// `reasoning` OBJECT's deeper normalization (`effort`/`enabled` aliases) is
/// NOT mirrored — it is withheld from forwarding instead (same predicate), and
/// only degrades routing here.
pub fn resolve_render_opts(request: &serde_json::Value) -> RenderOpts {
    let thinking = match request
        .get("chat_template_kwargs")
        .and_then(|k| k.as_object())
        .and_then(|o| o.get("thinking"))
    {
        Some(v) => json_truthy(v),
        None => match request.get("reasoning_effort").filter(|e| !e.is_null()) {
            Some(e) => !matches!(e, serde_json::Value::String(s) if s == "none"),
            None => default_thinking(),
        },
    };

    // Effective effort source, mirroring the engine's ctk-pop-with-precedence.
    let effort_field = request
        .get("chat_template_kwargs")
        .and_then(|k| k.as_object())
        .and_then(|o| o.get("reasoning_effort"))
        .filter(|v| !v.is_null())
        .or_else(|| request.get("reasoning_effort"));
    let reasoning_effort = match effort_field {
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "max" => ReasoningEffort::Max,
            "high" => ReasoningEffort::High,
            _ => ReasoningEffort::None,
        },
        // Absent or explicit null (both channels): the engine consults env.
        Some(serde_json::Value::Null) | None => default_effort_enum(),
        // Any other present value (numeric, bool, …): never `max`/`high`, and
        // the env default is NOT consulted.
        Some(_) => ReasoningEffort::None,
    };

    RenderOpts {
        thinking,
        reasoning_effort,
    }
}

/// The env default resolved to an enum (only `max`/`high` matter).
fn default_effort_enum() -> ReasoningEffort {
    match default_reasoning_effort().as_deref() {
        Some("max") => ReasoningEffort::Max,
        Some("high") => ReasoningEffort::High,
        _ => ReasoningEffort::None,
    }
}

/// Python-truthiness of a JSON value, mirroring the engine's `if thinking_requested`
/// test on the raw `chat_template_kwargs.thinking` value (`serving_chat.py`): a bool
/// as-is, a string/array/object truthy iff non-empty, a number iff non-zero, null
/// falsy. A conformant client sends a bool; this only matters for odd payloads, and
/// keeps routing matching the engine on them.
fn json_truthy(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::String(s) => !s.is_empty(),
        serde_json::Value::Number(n) => n.as_f64().is_none_or(|f| f != 0.0),
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
        serde_json::Value::Null => false,
    }
}

/// Parse a boolean env value the way the engine's `EnvBool` does (`environ.py`):
/// case-insensitive `true`/`1`/`yes`/`y` → `Some(true)`, `false`/`0`/`no`/`n` →
/// `Some(false)`, anything else `None`. Matching the engine's exact token set is
/// load-bearing: `y` is engine-true, so a set that omitted it would silently
/// disagree with a `SGLANG_DEFAULT_THINKING=y` engine.
fn parse_env_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "y" => Some(true),
        "false" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

/// Resolve the thinking default from the env value (pure; the env read + one-time
/// caching + logging live in [`router_defaults`]). Mirrors the engine's
/// `EnvBool` + `EnvField.get`: a recognized token → that bool; a non-empty
/// unrecognized value → WARN + `false` — the engine behaves identically (its
/// `EnvField.get` catches `EnvBool.parse`'s `ValueError`, warns, and returns the
/// `False` default; it does NOT surface a hard error). Unset/empty → `false`.
fn resolve_default_thinking(env: Option<&str>) -> bool {
    match env {
        Some(s) if !s.is_empty() => parse_env_bool(s).unwrap_or_else(|| {
            tracing::warn!(
                value = %s,
                "SGLANG_ROUTER_DSV4_DEFAULT_THINKING is not a recognized boolean \
                 (true/1/yes/y | false/0/no/n); using false (the engine warns + defaults \
                 false the same way) — dsv4 routing renders chat mode, mismatching a thinking engine"
            );
            false
        }),
        _ => false,
    }
}

/// Resolve the reasoning-effort default from the env value (pure). Only `max`/`high`
/// affect rendering; anything else collapses to `None` silently — matching the
/// engine, which keeps only `max`/`high` (`serving_chat.py`) and does not warn.
fn resolve_default_effort(env: Option<&str>) -> Option<String> {
    match env {
        Some("max") => Some("max".to_owned()),
        Some("high") => Some("high".to_owned()),
        _ => None,
    }
}

/// The router's render defaults, resolved once from env
/// (`SGLANG_ROUTER_DSV4_DEFAULT_THINKING` / `SGLANG_ROUTER_DSV4_REASONING_EFFORT`)
/// and logged once at INFO so an operator can confirm they match the engine's
/// `SGLANG_DEFAULT_THINKING` / `SGLANG_DSV4_REASONING_EFFORT`. Both are logged
/// together because an effort mismatch is a LARGER divergence than a thinking one
/// (the `max` preamble sits at block 0, so a mismatch shifts every block hash, not
/// just the trailing token) — it must be equally visible.
fn router_defaults() -> &'static (bool, Option<String>) {
    static V: std::sync::OnceLock<(bool, Option<String>)> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        let thinking = resolve_default_thinking(
            std::env::var("SGLANG_ROUTER_DSV4_DEFAULT_THINKING")
                .ok()
                .as_deref(),
        );
        let effort = resolve_default_effort(
            std::env::var("SGLANG_ROUTER_DSV4_REASONING_EFFORT")
                .ok()
                .as_deref(),
        );
        tracing::info!(
            default_thinking = thinking,
            default_reasoning_effort = effort.as_deref().unwrap_or("(none)"),
            "dsv4 router render defaults resolved; must match the engine's SGLANG_DEFAULT_THINKING \
             / SGLANG_DSV4_REASONING_EFFORT (i.e. --default-chat-template-kwargs) for cache-aware \
             routing to match"
        );
        (thinking, effort)
    })
}

fn default_thinking() -> bool {
    router_defaults().0
}

fn default_reasoning_effort() -> &'static Option<String> {
    &router_defaults().1
}

/// The `encoding_dsv4.REASONING_EFFORT_MAX` preamble, emitted at the very front
/// of the prompt (after BOS, before the system content) only in thinking mode
/// with `reasoning_effort == Max`. Kept byte-identical to the engine. NOTE:
/// newer engine builds split effort rendering into PREVIEW and OFFICIAL
/// profiles (only preview exists in the deployed build this constant is
/// pinned against); under the official profile `high` emits a preamble and
/// `max` uses different text — an engine upgrade to such a build must be
/// re-mirrored here (see the deploy note in `input_ids_safe_to_forward_dsv4`).
const REASONING_EFFORT_MAX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\nExplicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// Render `messages` (+ the request's top-level `tools`) into the DeepSeek-V4
/// chat prompt for routing.
///
/// Mirrors `encoding_dsv4.encode_messages` for the routing subset (chat and
/// thinking mode, text content), including tool calls, tool results, and per-turn
/// reasoning. `messages` is the request's `messages` array; `tools` is the
/// request's top-level `tools` array (OpenAI format), or `None`; `opts` carries
/// the resolved thinking mode / reasoning effort. Non-array `messages` renders to
/// just the BOS marker (the caller then tokenizes it and, finding no useful
/// prefix, degrades to min-load like any short prompt).
///
/// Byte-exactness with the engine is what lets a request's block hashes match
/// the engine's cached blocks. These line up with `encoding_dsv4`: tools
/// render right after the system content (where `serving_chat` attaches
/// `request.tools`); a `tool` message folds into the following user turn as a
/// `<tool_result>` block (`merge_tool_messages`); an assistant turn's
/// `tool_calls` render as a `DSML` block; multiple tool results in one turn are
/// ordered by their originating call (`sort_tool_results_by_call_order`); and in
/// thinking mode the `<think>` transitions and prior-turn `reasoning_content`
/// render per the engine's `drop_thinking` rules (see [`render_one`],
/// [`drop_thinking_messages`]). Request-level engine behaviors on top of this
/// (`task` attachment, trailing-assistant surgery) live in [`render_request`].
pub fn render_messages(
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    opts: RenderOpts,
) -> String {
    let raw = messages.as_array().map(Vec::as_slice).unwrap_or(&[]);
    let mut msgs = merge_tool_messages(raw);

    // The engine inserts an empty system message when the first message isn't a
    // system message; it renders to nothing but keeps the index logic aligned
    // (and is where tools attach). Doing it after the merge is equivalent — a
    // system turn never joins a user run.
    if msgs.first().map(|m| m.role != "system").unwrap_or(true) {
        msgs.insert(0, MergedMsg::plain("system"));
    }

    sort_tool_results_by_call_order(&mut msgs);

    // The engine attaches `request.tools` to `messages[0]` (the system message,
    // always present after the insertion above) and renders them immediately
    // after the system content. An empty `tools` array is falsy engine-side, so
    // treat it as no tools.
    let tool_list = tools
        .and_then(|t| t.as_array())
        .filter(|arr| !arr.is_empty());

    // Mirror `encode_messages`' drop resolution: the engine calls it with the
    // default `drop_thinking = true`, then forces it OFF when any message carries
    // tools (`effective_drop_thinking`). So: no tools → drop earlier turns'
    // reasoning; tools present → keep it (DeepSeek requires prior reasoning in the
    // context of a tool-calling multi-turn conversation).
    let effective_drop_thinking = tool_list.is_none();
    if opts.thinking && effective_drop_thinking {
        msgs = drop_thinking_messages(msgs);
    }
    let last_user_idx = last_user_index(&msgs);

    let mut out = String::from(BOS);
    // Reasoning-effort preamble sits at the very front (after BOS, before the
    // system content), thinking mode + `max` only — `encoding_dsv4.render_message`
    // index 0.
    if opts.thinking && opts.reasoning_effort == ReasoningEffort::Max {
        out.push_str(REASONING_EFFORT_MAX);
    }
    for i in 0..msgs.len() {
        render_one(
            i,
            &msgs,
            &mut out,
            opts,
            last_user_idx,
            effective_drop_thinking,
        );
        if i == 0 {
            if let Some(list) = tool_list {
                out.push_str("\n\n");
                out.push_str(&render_tools(list));
            }
        }
    }
    out
}

/// Index of the last `user`/`developer` message (the engine's
/// `find_last_user_index`), or `-1` when there is none. `i64` mirrors the
/// engine's `-1` sentinel so the `index >= last_user_idx` transition comparisons
/// are exact.
fn last_user_index(msgs: &[MergedMsg]) -> i64 {
    for i in (0..msgs.len()).rev() {
        if matches!(msgs[i].role.as_str(), "user" | "developer") {
            return i as i64;
        }
    }
    -1
}

/// Mirror `encoding_dsv4._drop_thinking_messages` (applied only in thinking mode
/// when dropping is in effect, i.e. no tools): messages at/after the last user
/// turn are kept verbatim; before it, `user`/`system`/`tool`/`latest_reminder`/
/// `direct_search_results` pass through, an assistant turn keeps everything but its
/// `reasoning_content`, and a `developer` (or any other) turn is dropped entirely.
fn drop_thinking_messages(msgs: Vec<MergedMsg>) -> Vec<MergedMsg> {
    let last_user = last_user_index(&msgs);
    let mut out = Vec::with_capacity(msgs.len());
    for (idx, mut m) in msgs.into_iter().enumerate() {
        let keep = matches!(
            m.role.as_str(),
            "user" | "system" | "tool" | "latest_reminder" | "direct_search_results"
        );
        if keep || idx as i64 >= last_user {
            out.push(m);
        } else if m.role == "assistant" {
            m.reasoning_content.clear();
            out.push(m);
        }
        // developer / other roles before the last user turn are dropped.
    }
    out
}

/// A tool call on an assistant turn, in the fields DSML rendering needs.
struct ToolCall {
    name: String,
    /// The OpenAI `arguments` — spec'd as a JSON string, but the type also
    /// permits an inlined object — kept raw and decoded at render time by
    /// [`encode_arguments_to_dsml`] (which mirrors the engine's `json.loads` +
    /// wrap-on-failure).
    arguments: serde_json::Value,
    /// OpenAI `id` (falling back to `function.id`); orders tool results.
    id: String,
}

/// One piece of a merged user turn: literal text, or a folded-in tool result.
enum Block {
    Text(String),
    ToolResult {
        content: String,
        tool_use_id: String,
    },
}

/// A message after `merge_tool_messages`. `blocks` is `Some` only for user turns
/// (their text + folded-in tool results); `tool_calls` is non-empty only for
/// assistant turns; `reasoning_content` is a prior assistant turn's thinking
/// block, rendered only in thinking mode (see [`render_one`]). `task` is the
/// engine's quick-instruction task key (request-attached or message-level),
/// preserved through the merge the way `encoding_dsv4.merge_tool_messages`
/// preserves it.
struct MergedMsg {
    role: String,
    content: String,
    tool_calls: Vec<ToolCall>,
    blocks: Option<Vec<Block>>,
    reasoning_content: String,
    task: Option<String>,
}

impl MergedMsg {
    fn plain(role: &str) -> Self {
        MergedMsg {
            role: role.to_string(),
            content: String::new(),
            tool_calls: Vec::new(),
            blocks: None,
            reasoning_content: String::new(),
            task: None,
        }
    }
}

/// Index of the trailing message iff it is a user turn a following USER
/// message should merge into: carrying blocks and with NO task — the
/// engine's user-merge guard (`merged[-1].get("task") is None`), which exists
/// only on the user fold: a task-carrying user turn terminates the run so
/// the following user message starts fresh (keeping its own keys).
fn open_user_idx(merged: &[MergedMsg]) -> Option<usize> {
    match merged.last() {
        Some(last) if last.role == "user" && last.blocks.is_some() && last.task.is_none() => {
            Some(merged.len() - 1)
        }
        _ => None,
    }
}

/// Index of the trailing message iff it is a user turn a TOOL message should
/// fold into: carrying blocks — the engine's tool fold has NO task guard
/// (`merged[-1].get("role") == "user" and "content_blocks" in merged[-1]`),
/// so a tool result folds into a task-carrying run too.
fn open_tool_fold_idx(merged: &[MergedMsg]) -> Option<usize> {
    match merged.last() {
        Some(last) if last.role == "user" && last.blocks.is_some() => Some(merged.len() - 1),
        _ => None,
    }
}

/// Mirror `encoding_dsv4.merge_tool_messages`: DeepSeek-V4 has no standalone
/// `tool` role, so a tool message folds into the preceding (or a fresh) user
/// turn as a `tool_result` block, and consecutive user turns coalesce into one
/// with a text block each. Other roles pass through unchanged. Message-level
/// `task` keys are preserved the way the engine preserves extra fields.
fn merge_tool_messages(raw: &[serde_json::Value]) -> Vec<MergedMsg> {
    let mut merged: Vec<MergedMsg> = Vec::with_capacity(raw.len());
    for m in raw {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        match role {
            "tool" => {
                let block = Block::ToolResult {
                    content: tool_result_content(m.get("content")),
                    tool_use_id: str_field(m, "tool_call_id"),
                };
                match open_tool_fold_idx(&merged) {
                    Some(idx) => merged[idx].blocks.as_mut().unwrap().push(block),
                    None => {
                        let mut um = MergedMsg::plain("user");
                        um.blocks = Some(vec![block]);
                        merged.push(um);
                    }
                }
            }
            "user" => {
                let text = content_to_string(m.get("content"));
                match open_user_idx(&merged) {
                    Some(idx) => merged[idx].blocks.as_mut().unwrap().push(Block::Text(text)),
                    None => {
                        let mut um = MergedMsg::plain("user");
                        um.content = text.clone();
                        um.blocks = Some(vec![Block::Text(text)]);
                        um.task = str_field_opt(m, "task");
                        merged.push(um);
                    }
                }
            }
            "assistant" => {
                let mut am = MergedMsg::plain("assistant");
                am.content = content_to_string(m.get("content"));
                am.tool_calls = parse_tool_calls(m.get("tool_calls"));
                // Prior-turn thinking block. The engine renders it verbatim as
                // `reasoning_content or ""` (a plain string), so pass it through
                // as-is; a missing/non-string value is treated as empty.
                am.reasoning_content = str_field(m, "reasoning_content");
                am.task = str_field_opt(m, "task");
                merged.push(am);
            }
            other => {
                let mut om = MergedMsg::plain(other);
                om.content = content_to_string(m.get("content"));
                om.task = str_field_opt(m, "task");
                merged.push(om);
            }
        }
    }
    merged
}

/// Extract an assistant turn's `tool_calls` into [`ToolCall`]s. Missing fields
/// degrade to empty rather than dropping the call — a malformed call still
/// contributes stable bytes to the hash. `id` falls back to `function.id` when
/// absent at the top level, matching the engine's sort-key extraction
/// (`tc.get("id") or tc.get("function",{}).get("id")`).
fn parse_tool_calls(v: Option<&serde_json::Value>) -> Vec<ToolCall> {
    let Some(arr) = v.and_then(|t| t.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .map(|tc| {
            let func = tc.get("function");
            let id = {
                let top = str_field(tc, "id");
                if top.is_empty() {
                    func.map(|f| str_field(f, "id")).unwrap_or_default()
                } else {
                    top
                }
            };
            ToolCall {
                name: func.map(|f| str_field(f, "name")).unwrap_or_default(),
                // Keep the raw value (string or inlined object);
                // `encode_arguments_to_dsml` reproduces the engine's handling of
                // both.
                arguments: func
                    .and_then(|f| f.get("arguments"))
                    .cloned()
                    .unwrap_or(serde_json::Value::Null),
                id,
            }
        })
        .collect()
}

/// Mirror `encoding_dsv4.sort_tool_results_by_call_order`: when a user turn
/// carries more than one tool result, order them by the position of their
/// originating call in the most recent assistant turn's `tool_calls`. Text
/// blocks keep their slots; only the tool-result slots are reordered among
/// themselves. A single tool result (or no preceding calls) is left untouched.
fn sort_tool_results_by_call_order(merged: &mut [MergedMsg]) {
    let mut order: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in merged.iter_mut() {
        if m.role == "assistant" && !m.tool_calls.is_empty() {
            order.clear();
            for (idx, tc) in m.tool_calls.iter().enumerate() {
                if !tc.id.is_empty() {
                    order.insert(tc.id.clone(), idx);
                }
            }
            continue;
        }
        let Some(blocks) = m.blocks.as_mut() else {
            continue;
        };
        // Capture the tool-result slot positions FIRST: the extraction below
        // swaps a `Text` placeholder into each, so a later `matches!(ToolResult)`
        // would no longer find them.
        let slots: Vec<usize> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| matches!(b, Block::ToolResult { .. }))
            .map(|(idx, _)| idx)
            .collect();
        if slots.len() <= 1 || order.is_empty() {
            continue;
        }
        // Pull the tool-result blocks out, stable-sort by call order (unknown
        // ids sort to 0, matching the engine's `.get(id, 0)`), drop them back
        // into the same slots — text blocks keep their positions.
        let mut results: Vec<Block> = slots
            .iter()
            .map(|&idx| std::mem::replace(&mut blocks[idx], Block::Text(String::new())))
            .collect();
        results.sort_by_key(|b| match b {
            Block::ToolResult { tool_use_id, .. } => *order.get(tool_use_id).unwrap_or(&0),
            Block::Text(_) => 0,
        });
        for (slot, b) in slots.into_iter().zip(results) {
            blocks[slot] = b;
        }
    }
}

/// Append merged message `i`'s encoded form to `out`.
fn render_one(
    i: usize,
    msgs: &[MergedMsg],
    out: &mut String,
    opts: RenderOpts,
    last_user_idx: i64,
    effective_drop_thinking: bool,
) {
    let m = &msgs[i];
    match m.role.as_str() {
        "system" => out.push_str(&m.content),
        "user" | "developer" => {
            out.push_str(USER);
            match &m.blocks {
                // A merged user turn renders its blocks joined by `\n\n`.
                Some(blocks) => {
                    let parts: Vec<String> = blocks.iter().map(render_block).collect();
                    out.push_str(&parts.join("\n\n"));
                }
                // `developer` (and any user that never went through the merge)
                // renders its bare content.
                None => out.push_str(&m.content),
            }
        }
        "assistant" => {
            // Thinking mode renders the turn's `reasoning_content` followed by
            // `</think>` before its content; the opening `<think>` came from the
            // preceding user turn's transition below. Kept when reasoning isn't
            // being dropped (tools present) OR this turn is strictly after the last
            // user turn — matching `encoding_dsv4.render_message`, including its
            // `prev_has_task` rule: a task on the PRECEDING message marks this
            // turn as a task output, whose thinking is never rendered.
            let prev_has_task = i > 0 && msgs[i - 1].task.is_some();
            if opts.thinking
                && !prev_has_task
                && (!effective_drop_thinking || i as i64 > last_user_idx)
            {
                out.push_str(&m.reasoning_content);
                out.push_str(THINK_END);
            }
            out.push_str(&m.content);
            if !m.tool_calls.is_empty() {
                out.push_str("\n\n");
                out.push_str(&render_tool_calls(&m.tool_calls));
            }
            out.push_str(EOS);
        }
        // Unknown roles aren't part of routing traffic; emit the content so a
        // stray role still contributes something rather than vanishing.
        _ => out.push_str(&m.content),
    }

    // Generation-prompt / task transition. The engine appends it only when this
    // is the last message OR the next message is an assistant/reminder turn.
    let next_takes_transition = match msgs.get(i + 1) {
        Some(next) => next.role == "assistant" || next.role == "latest_reminder",
        None => true,
    };
    if !next_takes_transition {
        return;
    }
    if let Some(task) = m.task.as_deref() {
        // Task transition (`encoding_dsv4.render_message`): any non-`action`
        // task appends its special token directly — without the assistant
        // opening — while `action` opens an assistant turn first. Task names
        // are validated in `render_request`; plain-render callers must
        // pre-gate instead (the cache-sim shape gate does).
        let sp = task_sp_token(task).expect("task validated by the caller's gate");
        if task != "action" {
            out.push_str(sp);
        } else {
            out.push_str(ASSISTANT);
            out.push_str(if opts.thinking {
                THINK_START
            } else {
                THINK_END
            });
            out.push_str(sp);
        }
    } else if m.role == "user" || m.role == "developer" {
        out.push_str(ASSISTANT);
        // Chat mode always closes with `</think>`. Thinking mode opens the next
        // assistant turn with `<think>`: always when reasoning is kept (tools
        // present), else only at/after the last user turn (the current
        // generation) — mirroring `encoding_dsv4.render_message`.
        let token = if opts.thinking && (!effective_drop_thinking || i as i64 >= last_user_idx) {
            THINK_START
        } else {
            THINK_END
        };
        out.push_str(token);
    }
}

/// Render one merged-user block: text verbatim, a tool result wrapped in the
/// engine's `<tool_result>…</tool_result>` (`tool_output_template`).
fn render_block(b: &Block) -> String {
    match b {
        Block::Text(t) => t.clone(),
        Block::ToolResult { content, .. } => format!("<tool_result>{content}</tool_result>"),
    }
}

/// Render an assistant turn's tool calls as the engine's DSML block
/// (`tool_calls_template` wrapping one `tool_call_template` per call).
fn render_tool_calls(tool_calls: &[ToolCall]) -> String {
    let invokes = tool_calls
        .iter()
        .map(|tc| {
            format!(
                "<{DSML}invoke name=\"{}\">\n{}\n</{DSML}invoke>",
                tc.name,
                encode_arguments_to_dsml(&tc.arguments)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!("<{DSML}tool_calls>\n{invokes}\n</{DSML}tool_calls>")
}

/// Encode a tool call's `arguments` into DSML `<parameter>` lines, mirroring
/// `encoding_dsv4.encode_arguments_to_dsml` for the case that matters: a valid
/// JSON-object string, whose keys each become a param — a string value raw with
/// `string="true"`, anything else the Python-`json.dumps` form with
/// `string="false"`. The other shapes are degenerate — an inlined object, an
/// unparsable string, or a parsed non-object — which a conformant client never
/// sends and the engine rejects (its exact error path varies by version), so the
/// router handles them defensively rather than reproducing that error: a
/// non-string value or unparsable string is wrapped as a single
/// `{"arguments": …}` param, and a parsed non-object yields none.
fn encode_arguments_to_dsml(arguments: &serde_json::Value) -> String {
    let parsed = match arguments {
        serde_json::Value::String(s) => serde_json::from_str::<serde_json::Value>(s)
            .unwrap_or_else(|_| serde_json::json!({ "arguments": s })),
        other => serde_json::json!({ "arguments": other }),
    };
    let Some(obj) = parsed.as_object() else {
        return String::new();
    };
    obj.iter()
        .map(|(k, v)| {
            let (is_str, value) = match v {
                serde_json::Value::String(s) => ("true", s.clone()),
                _ => ("false", py_json(v)),
            };
            format!("<{DSML}parameter name=\"{k}\" string=\"{is_str}\">{value}</{DSML}parameter>")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Flatten a `tool` message's `content` for a `<tool_result>` body, mirroring
/// the engine's pre-merge flatten on the dsv4 path
/// (`process_content_for_template_format(_, "string")` runs over EVERY client
/// message, tool messages included): a string as-is; an array to its text
/// parts joined with a single space, non-text parts dropped. (The
/// `[Unsupported <type>]` shape seen in `encoding_dsv4` only applies to
/// engine-internal `content_blocks`, which clients cannot send.)
fn tool_result_content(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// A message field as an optional owned string (`None` when absent/non-string).
fn str_field_opt(m: &serde_json::Value, key: &str) -> Option<String> {
    m.get(key).and_then(|v| v.as_str()).map(str::to_owned)
}

/// A message field as an owned string, empty when absent/non-string.
fn str_field(m: &serde_json::Value, key: &str) -> String {
    m.get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Flatten a message `content` field to a string: a plain string as-is; an
/// OpenAI parts array to its `type == "text"` parts joined with a single space
/// (mirroring `process_content_for_template_format(_, "string")`, which the
/// engine applies to dsv4 before encoding and which ignores non-text parts);
/// a NUMBER coerced to string (pydantic's lax `Union[str, List[...], None]`
/// coerces ints/floats before the engine ever renders); anything else
/// (bool/object — the engine 422s those) to empty.
fn content_to_string(content: Option<&serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Number(n)) => n.to_string(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"))
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Fixed tools-section text from the engine's `encoding_dsv4.TOOLS_TEMPLATE`
/// with the constant tokens (`dsml_token`, thinking start/end) already
/// substituted; the tool schemas are the only variable part and slot between
/// `TOOLS_PREFIX` and `TOOLS_SUFFIX`. Kept byte-identical to the engine so the
/// router's block hashes match its cached blocks.
const TOOLS_PREFIX: &str = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n";
const TOOLS_SUFFIX: &str =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

/// Render the request's top-level OpenAI `tools` array into the DeepSeek-V4
/// tools section, mirroring `encoding_dsv4.render_tools`: each tool's canonical
/// `function` (see [`canonical_function`]) is serialized Python-style and the
/// schemas are joined with `\n`, between the fixed preamble and trailer.
fn render_tools(tools: &[serde_json::Value]) -> String {
    let schemas = tools
        .iter()
        .map(|t| py_json(&canonical_function(t)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{TOOLS_PREFIX}{schemas}{TOOLS_SUFFIX}")
}

/// Reproduce the engine's `Function.model_dump()` (`serving_chat`) for one
/// OpenAI tool: take its `function` object and re-emit exactly
/// `description, name, parameters, strict` (plus `defer_loading` only when set),
/// injecting the pydantic defaults for the optional fields (`description` and
/// `parameters` → null, `strict` → false) and dropping unknown fields. `name`
/// has no engine-side default (a missing one is a 422), so an absent name emits
/// `null` here rather than being "defaulted". A tool-level `defer_loading` (a
/// GLM extension the engine propagates into the function) is out of scope — only
/// `function.defer_loading` is read. This canonical shape — not the raw client
/// object — is what the engine serializes, so the router must key on the same
/// bytes.
fn canonical_function(tool: &serde_json::Value) -> serde_json::Value {
    let func = tool.get("function");
    let field = |k: &str| func.and_then(|f| f.get(k)).cloned();
    let mut m = serde_json::Map::new();
    m.insert(
        "description".to_string(),
        field("description").unwrap_or(serde_json::Value::Null),
    );
    m.insert(
        "name".to_string(),
        field("name").unwrap_or(serde_json::Value::Null),
    );
    m.insert(
        "parameters".to_string(),
        field("parameters").unwrap_or(serde_json::Value::Null),
    );
    m.insert(
        "strict".to_string(),
        field("strict").unwrap_or(serde_json::Value::Bool(false)),
    );
    if let Some(dl) = field("defer_loading") {
        if !dl.is_null() {
            m.insert("defer_loading".to_string(), dl);
        }
    }
    serde_json::Value::Object(m)
}

/// Serialize `value` the way Python's `json.dumps(v, ensure_ascii=False)` does:
/// `", "` between elements/members and `": "` after each key (serde's compact
/// form omits those spaces, changing the bytes the engine's block hashes key
/// on). Key order and non-ASCII pass through unchanged (`ensure_ascii=False` +
/// the crate's `preserve_order` feature).
///
/// CAVEAT: number formatting matches Python for the ints/floats/bools that
/// appear in real tool schemas, but not universally — scientific-notation floats
/// (`1e-5` renders `0.00001` here vs Python `1e-05`) and integers beyond f64's
/// exact range diverge from Python's `repr`. Such a value in a tool schema would
/// miss the exact cache match and degrade to min-load; none appear in observed
/// (OpenCode) tool schemas, so this is left as a known edge rather than
/// reimplementing Python float formatting.
fn py_json(value: &serde_json::Value) -> String {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, PyJsonFormatter);
    value
        .serialize(&mut serializer)
        .expect("serializing a serde_json::Value into a Vec is infallible");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// serde_json formatter emitting Python `json.dumps` default separators
/// (`", "` / `": "`) instead of serde's compact `,` / `:`.
struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn effort(s: Option<&str>) -> ReasoningEffort {
        match s {
            Some("max") => ReasoningEffort::Max,
            Some("high") => ReasoningEffort::High,
            _ => ReasoningEffort::None,
        }
    }

    /// Byte-exact parity against the engine's `encoding_dsv4.encode_messages` in
    /// BOTH chat and thinking mode, across fixtures generated from the engine
    /// encoder itself (transition token `<think>`/`</think>`, prior reasoning
    /// kept-with-tools vs dropped-without, the empty-reasoning `<think></think>`
    /// block, and the `reasoning_effort=max` front preamble). A mismatch here is
    /// exactly a router↔engine routing-tokenization divergence — the thing that
    /// collapses cache-aware routing on a thinking-mode engine.
    #[test]
    fn thinking_and_chat_parity_fixtures() {
        let raw = include_str!("testdata/dsv4_thinking_cases.json");
        let cases: Vec<serde_json::Value> = serde_json::from_str(raw).expect("fixture json parses");
        assert!(!cases.is_empty(), "fixtures present");
        for c in &cases {
            let name = c["name"].as_str().unwrap();
            let tools = c.get("tools").filter(|t| !t.is_null());
            let opts = RenderOpts {
                thinking: c["thinking"].as_bool().unwrap(),
                reasoning_effort: effort(c["reasoning_effort"].as_str()),
            };
            let got = render_messages(&c["messages"], tools, opts);
            assert_eq!(got, c["expected"].as_str().unwrap(), "case `{name}`");
        }
    }

    /// Byte-exact against the engine's `/tokenize`: a single user turn renders
    /// `BOS <｜User｜> content <｜Assistant｜> </think>`.
    #[test]
    fn single_user_turn() {
        let out = render_messages(
            &json!([{"role":"user","content":"ABCD"}]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// A leading system message renders as bare content (no marker), before the
    /// user turn.
    #[test]
    fn system_then_user() {
        let out = render_messages(
            &json!([
                {"role":"system","content":"SYS"},
                {"role":"user","content":"ABCD"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜>SYS<｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// Multi-turn: each prior user turn gets the generation prompt, the prior
    /// assistant turn is closed by EOS. Matches the engine token stream
    /// `[0,128803,55,19,128804,128822,35,19,1,128803,55,20,128804,128822]`.
    #[test]
    fn multi_turn() {
        let out = render_messages(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A1"},
                {"role":"user","content":"U2"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1<｜Assistant｜></think>A1<｜end▁of▁sentence｜><｜User｜>U2<｜Assistant｜></think>"
        );
    }

    /// An empty leading system message (already present) is not duplicated and
    /// renders to nothing — same result as a bare user turn.
    #[test]
    fn explicit_empty_system_is_not_duplicated() {
        let out = render_messages(
            &json!([
                {"role":"system","content":""},
                {"role":"user","content":"ABCD"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>ABCD<｜Assistant｜></think>"
        );
    }

    /// Multi-part text content flattens to its `type == "text"` parts joined
    /// with a single space (mirroring the engine's
    /// `process_content_for_template_format(_, "string")`), NOT concatenated.
    #[test]
    fn array_content_flattens_text_parts() {
        let out = render_messages(
            &json!([
                {"role":"user","content":[{"type":"text","text":"AB"},{"type":"text","text":"CD"}]}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>AB CD<｜Assistant｜></think>"
        );
    }

    /// Consecutive user turns merge into one `<｜User｜>` turn joined with `\n\n`
    /// (the engine's `merge_tool_messages`), so only one user marker and one
    /// generation prompt are emitted — not a marker per message.
    #[test]
    fn consecutive_user_turns_merge() {
        let out = render_messages(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"user","content":"U2"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1\n\nU2<｜Assistant｜></think>"
        );
    }

    /// A run of user turns split by an assistant turn does NOT merge across the
    /// assistant: each side is its own user turn.
    #[test]
    fn user_runs_do_not_merge_across_assistant() {
        let out = render_messages(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"user","content":"U2"},
                {"role":"assistant","content":"A1"},
                {"role":"user","content":"U3"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            out,
            "<｜begin▁of▁sentence｜><｜User｜>U1\n\nU2<｜Assistant｜></think>A1<｜end▁of▁sentence｜><｜User｜>U3<｜Assistant｜></think>"
        );
    }

    /// A `developer` turn renders identically to a user turn for text content
    /// (the engine nests the same `<｜User｜>` marker) and takes the generation
    /// prompt. Developer turns are not merged (only `user` runs merge), so two
    /// developers emit two markers.
    #[test]
    fn developer_role_renders_like_user_without_merging() {
        assert_eq!(
            render_messages(
                &json!([{"role":"developer","content":"D1"}]),
                None,
                RenderOpts::chat()
            ),
            "<｜begin▁of▁sentence｜><｜User｜>D1<｜Assistant｜></think>"
        );
        assert_eq!(
            render_messages(
                &json!([
                    {"role":"developer","content":"D1"},
                    {"role":"developer","content":"D2"}
                ]),
                None,
                RenderOpts::chat()
            ),
            "<｜begin▁of▁sentence｜><｜User｜>D1<｜User｜>D2<｜Assistant｜></think>"
        );
    }

    /// An empty messages list renders to just the BOS marker — the documented
    /// degrade path (the caller then routes by min-load on the empty prefix).
    #[test]
    fn empty_messages_renders_bos_only() {
        assert_eq!(
            render_messages(&json!([]), None, RenderOpts::chat()),
            "<｜begin▁of▁sentence｜>"
        );
    }

    /// `py_json` reproduces Python `json.dumps(v, ensure_ascii=False)` default
    /// separators (`", "` / `": "`) and preserves key order — the exact bytes the
    /// engine's `to_json` produces (serde's compact form would drop the spaces).
    #[test]
    fn py_json_uses_python_separators_and_preserves_order() {
        let v = json!({"b": 1, "a": [1, 2], "c": {"x": true}});
        assert_eq!(py_json(&v), r#"{"b": 1, "a": [1, 2], "c": {"x": true}}"#);
    }

    /// A tool's `function` is re-emitted as the engine's `Function.model_dump`:
    /// fixed order (description, name, parameters, strict), pydantic defaults
    /// injected for omitted fields, unknown fields dropped, then Python-serialized.
    #[test]
    fn canonical_function_reorders_and_injects_defaults() {
        // Client sends keys out of order, omits description/strict, adds an extra.
        let tool = json!({
            "type": "function",
            "function": {"name": "ping", "parameters": {"type": "object"}, "x-extra": 1}
        });
        assert_eq!(
            py_json(&canonical_function(&tool)),
            r#"{"description": null, "name": "ping", "parameters": {"type": "object"}, "strict": false}"#
        );
    }

    /// Byte-exact against the engine (`encode_messages`, chat mode): tools render
    /// right after the system content, each `function` canonicalized + serialized
    /// Python-style. Pinned against `encoding_dsv4.encode_messages` output.
    #[test]
    fn renders_tools_after_system_content() {
        let messages = json!([
            {"role":"system","content":"SYS"},
            {"role":"user","content":"hi"}
        ]);
        let tools = json!([
            {"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}
        ]);
        let expected = "<｜begin▁of▁sentence｜>SYS\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"description\": \"Get weather\", \"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}, \"strict\": false}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>hi<｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, Some(&tools), RenderOpts::chat()),
            expected
        );
    }

    /// An empty `tools` array is falsy engine-side — no tools block is rendered.
    #[test]
    fn empty_tools_array_renders_no_tools_block() {
        let messages = json!([{"role":"user","content":"hi"}]);
        assert_eq!(
            render_messages(&messages, Some(&json!([])), RenderOpts::chat()),
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"
        );
    }

    /// With no system message the engine inserts an empty one and still renders
    /// the tools block (right after the empty system content, i.e. after BOS).
    #[test]
    fn renders_tools_when_no_system_message_present() {
        let messages = json!([{"role":"user","content":"hi"}]);
        let tools = json!([{"type":"function","function":{"name":"ping","description":"p"}}]);
        let out = render_messages(&messages, Some(&tools), RenderOpts::chat());
        assert!(
            out.starts_with("<｜begin▁of▁sentence｜>\n\n## Tools\n\n"),
            "tools block should follow the inserted empty system; got: {out}"
        );
        assert!(out.contains(
            r#"{"description": "p", "name": "ping", "parameters": null, "strict": false}"#
        ));
        assert!(out.ends_with("<｜User｜>hi<｜Assistant｜></think>"));
    }

    /// Byte-exact against the engine (`encode_messages`, chat mode): an assistant
    /// `tool_calls` turn renders the DSML block — string args raw with
    /// `string="true"`, others Python-serialized with `string="false"` — and a
    /// following `tool` message folds into the next user turn as a
    /// `<tool_result>` block.
    #[test]
    fn renders_assistant_tool_calls_and_tool_result() {
        let messages = json!([
            {"role":"user","content":"u1"},
            {"role":"assistant","content":"reading","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"read","arguments":"{\"filePath\": \"/x\", \"limit\": 10, \"nested\": {\"a\": 1}}"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"FILE"},
            {"role":"user","content":"u2"}
        ]);
        let expected = "<｜begin▁of▁sentence｜><｜User｜>u1<｜Assistant｜></think>reading\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"read\">\n<｜DSML｜parameter name=\"filePath\" string=\"true\">/x</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">10</｜DSML｜parameter>\n<｜DSML｜parameter name=\"nested\" string=\"false\">{\"a\": 1}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>FILE</tool_result>\n\nu2<｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: multiple tool results in one user turn are
    /// reordered to the preceding assistant's `tool_calls` order. Results arrive
    /// b,a; the calls were a,b → rendered a,b.
    #[test]
    fn tool_results_sorted_by_call_order() {
        let messages = json!([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"a","type":"function","function":{"name":"t1","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"t2","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":"RB"},
            {"role":"tool","tool_call_id":"a","content":"RA"}
        ]);
        let expected = "<｜begin▁of▁sentence｜>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"t1\">\n\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"t2\">\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>RA</tool_result>\n\n<tool_result>RB</tool_result><｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: multiple tools render one canonical schema
    /// per line (`\n`-joined). Tool `b` omits everything but `name` (defaults
    /// injected: description/parameters → null) and sets `strict: true`.
    #[test]
    fn renders_multiple_tools() {
        let messages = json!([
            {"role":"system","content":"S"},
            {"role":"user","content":"hi"}
        ]);
        let tools = json!([
            {"type":"function","function":{"name":"a","description":"da","parameters":{"type":"object"}}},
            {"type":"function","function":{"name":"b","strict":true}}
        ]);
        let expected = "<｜begin▁of▁sentence｜>S\n\n## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n...\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"description\": \"da\", \"name\": \"a\", \"parameters\": {\"type\": \"object\"}, \"strict\": false}\n{\"description\": null, \"name\": \"b\", \"parameters\": null, \"strict\": true}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n<｜User｜>hi<｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, Some(&tools), RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: an inlined-object `arguments` (permitted by
    /// the OpenAI type) wraps into a single `arguments` param, not expanded per
    /// key — mirroring the engine's `json.loads(<dict>)` TypeError → wrap.
    #[test]
    fn inlined_object_arguments_wrapped_like_engine() {
        let messages = json!([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":{"x":1,"y":"z"}}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"R"}
        ]);
        let expected = "<｜begin▁of▁sentence｜>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"arguments\" string=\"false\">{\"x\": 1, \"y\": \"z\"}</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>R</tool_result><｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// Byte-exact against the engine: an unparsable `arguments` string wraps
    /// into a single `arguments` param carrying the raw string (`string="true"`),
    /// mirroring the engine's `json.loads` except-branch.
    #[test]
    fn invalid_json_arguments_wrapped_like_engine() {
        let messages = json!([
            {"role":"assistant","content":"","tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":"not json"}}
            ]},
            {"role":"tool","tool_call_id":"c1","content":"R"}
        ]);
        let expected = "<｜begin▁of▁sentence｜>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"arguments\" string=\"true\">not json</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜><｜User｜><tool_result>R</tool_result><｜Assistant｜></think>";
        assert_eq!(
            render_messages(&messages, None, RenderOpts::chat()),
            expected
        );
    }

    /// `parse_env_bool` matches the engine's `EnvBool` token set exactly
    /// (`true/1/yes/y` | `false/0/no/n`, case-insensitive), and returns `None`
    /// for anything else — including `on`, which a hand-rolled set might wrongly
    /// accept and thereby diverge from the engine.
    #[test]
    fn parse_env_bool_matches_engine_token_set() {
        for t in ["true", "1", "yes", "y", "TRUE", "Yes", "Y"] {
            assert_eq!(parse_env_bool(t), Some(true), "{t}");
        }
        for f in ["false", "0", "no", "n", "FALSE", "No"] {
            assert_eq!(parse_env_bool(f), Some(false), "{f}");
        }
        for u in ["on", "off", "enabled", "t", "", "2"] {
            assert_eq!(parse_env_bool(u), None, "{u}");
        }
    }

    /// `json_truthy` mirrors Python truthiness on the raw `thinking` value.
    #[test]
    fn json_truthy_mirrors_python() {
        assert!(json_truthy(&json!(true)));
        assert!(!json_truthy(&json!(false)));
        assert!(json_truthy(&json!("true"))); // non-empty string is truthy
        assert!(json_truthy(&json!("false"))); // even "false" — non-empty
        assert!(!json_truthy(&json!("")));
        assert!(json_truthy(&json!(1)));
        assert!(!json_truthy(&json!(0)));
        assert!(!json_truthy(&serde_json::Value::Null));
        assert!(!json_truthy(&json!([])));
        assert!(json_truthy(&json!([1])));
        assert!(!json_truthy(&json!({})));
        assert!(json_truthy(&json!({ "k": 1 })));
    }

    /// `resolve_render_opts` honors per-request overrides (the paths that don't
    /// depend on env): `chat_template_kwargs.thinking` truthiness; `reasoning_effort`
    /// top-level ONLY (the engine's dsv4 branch never reads it from
    /// chat_template_kwargs); and the protocol validator's thinking default
    /// (effort present → `!= "none"`).
    #[test]
    fn resolve_render_opts_request_overrides() {
        let opts = resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": true}}));
        assert!(opts.thinking);
        assert_eq!(opts.reasoning_effort, ReasoningEffort::None);

        assert!(
            !resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": false}})).thinking
        );
        // non-bool thinking follows truthiness (matches the engine), not `.as_bool()`.
        assert!(
            resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": "true"}})).thinking
        );
        assert!(
            !resolve_render_opts(&json!({"chat_template_kwargs": {"thinking": null}})).thinking
        );

        // chat_template_kwargs.reasoning_effort is popped ONTO the top-level
        // field by `_convert_to_internal_request` before the dsv4 branch reads
        // it — ctk wins over a set top-level.
        assert_eq!(
            resolve_render_opts(&json!({
                "reasoning_effort": "high",
                "chat_template_kwargs": {"reasoning_effort": "max"}
            }))
            .reasoning_effort,
            ReasoningEffort::Max
        );
        // …but a present-null ctk value lets the top-level field through.
        assert_eq!(
            resolve_render_opts(&json!({
                "reasoning_effort": "high",
                "chat_template_kwargs": {"reasoning_effort": null}
            }))
            .reasoning_effort,
            ReasoningEffort::High
        );
        assert_eq!(
            resolve_render_opts(&json!({"reasoning_effort": "high"})).reasoning_effort,
            ReasoningEffort::High
        );
        // unknown effort collapses to None (matches the engine's max/high-only gate).
        assert_eq!(
            resolve_render_opts(&json!({"reasoning_effort": "medium"})).reasoning_effort,
            ReasoningEffort::None
        );
        // and thinking still defaults from effort != "none".
        assert!(resolve_render_opts(&json!({"reasoning_effort": "medium"})).thinking);
        // a present-null effort falls to the (false) env default for thinking…
        assert!(!resolve_render_opts(&json!({"reasoning_effort": null})).thinking);
        // …while an explicit null THINKING key wins over the effort default
        // (setdefault semantics: key presence, not truthiness).
        assert!(
            !resolve_render_opts(
                &json!({"chat_template_kwargs": {"thinking": null}, "reasoning_effort": "high"})
            )
            .thinking
        );
        // "none" effort force-disables thinking (validator: thinking = effort != "none").
        assert!(!resolve_render_opts(&json!({"reasoning_effort": "none"})).thinking);
        // numeric (token-budget) effort: thinking on, effort None, env not consulted.
        let o = resolve_render_opts(&json!({"reasoning_effort": 512}));
        assert!(o.thinking);
        assert_eq!(o.reasoning_effort, ReasoningEffort::None);
    }

    /// The env-default resolvers (the deploy-critical `SGLANG_ROUTER_DSV4_*` knobs):
    /// pure over the env value so the empty / unrecognized / valid branches are
    /// pinned without the process-global `OnceLock`.
    #[test]
    fn resolve_default_thinking_branches() {
        assert!(!resolve_default_thinking(None)); // unset → false (engine default)
        assert!(!resolve_default_thinking(Some(""))); // set-empty → false
        assert!(resolve_default_thinking(Some("true")));
        assert!(resolve_default_thinking(Some("y"))); // engine EnvBool token
        assert!(!resolve_default_thinking(Some("false")));
        assert!(!resolve_default_thinking(Some("enabled"))); // unrecognized → false (WARNs)
    }

    #[test]
    fn resolve_default_effort_branches() {
        assert_eq!(resolve_default_effort(None), None);
        assert_eq!(resolve_default_effort(Some("max")), Some("max".to_owned()));
        assert_eq!(
            resolve_default_effort(Some("high")),
            Some("high".to_owned())
        );
        assert_eq!(resolve_default_effort(Some("medium")), None); // engine keeps only max/high
        assert_eq!(resolve_default_effort(Some("")), None);
    }

    /// mirror `normalize_reasoning_inputs` + `_handle_last_assistant_message` cases.

    /// trailing assistant + continue_final_message=true: dropped; prefix back.
    /// flag unset: role rewritten to user; prefix none; generation prompt follows.
    /// engine bytes verified against serving_chat._handle_last_assistant_message.
    #[test]
    fn handle_trailing_assistant_both_flag_states() {
        let (text, prefix) = render_request(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A-prefix"}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts {
                task: None,
                continue_final_message: true,
            },
        )
        .unwrap();
        assert_eq!(prefix.as_deref(), Some("A-prefix"));
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>U1<｜Assistant｜></think>"
        );
        // The prefix lands AFTER that generation prompt at encode time
        // (see TokenizerRegistry::encode_chat's append), not before.

        // No surgery: trailing assistant rewritten to user, coalesced into the
        // preceding user run, then the generation prompt.
        let (text, prefix) = render_request(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A2"}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts::default(),
        )
        .unwrap();
        assert_eq!(prefix, None);
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>U1\n\nA2<｜Assistant｜></think>"
        );
        assert_eq!(
            text,
            render_messages(
                &json!([
                    {"role":"user","content":"U1"},
                    {"role":"user","content":"A2"}
                ]),
                None,
                RenderOpts::chat()
            ),
            "the rewrite must behave exactly like a client-sent user role"
        );
    }

    /// continue_final_message=true WITHOUT a trailing assistant is a no-op.
    #[test]
    fn continue_final_message_without_trailing_assistant_is_noop() {
        let (text, prefix) = render_request(
            &json!([{"role":"user","content":"hi"}]),
            None,
            RenderOpts::chat(),
            RequestParts {
                task: None,
                continue_final_message: true,
            },
        )
        .unwrap();
        assert_eq!(prefix, None);
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"
        );
    }

    /// The engine flattens content BEFORE the surgery, so array content with
    /// only text parts and null content both see the surgery — the flattened
    /// string is the prefix/rewrite content.
    #[test]
    fn trailing_assistant_flattened_content_shapes() {
        // all-text parts array, continue_final_message=true
        let (_, prefix) = render_request(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":[{"type":"text","text":"AB"},{"type":"text","text":"CD"}]}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts {
                task: None,
                continue_final_message: true,
            },
        )
        .unwrap();
        assert_eq!(prefix.as_deref(), Some("AB CD"));

        // null content: flattened to "" — surgery still runs.
        let (text, prefix) = render_request(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":null}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts {
                task: None,
                continue_final_message: true,
            },
        )
        .unwrap();
        assert_eq!(prefix.as_deref(), Some(""));
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>U1<｜Assistant｜></think>"
        );
    }

    /// The flag-false rewrite REPLACES the message wholesale, like the engine
    /// — a message-level `task` key on the trailing assistant must NOT
    /// survive, or the render would emit a task transition the engine never
    /// does.
    #[test]
    fn rewrite_drops_message_level_task_key() {
        let (text, _) = render_request(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A2","task":"query"}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts::default(),
        )
        .unwrap();
        assert!(
            !text.contains("<｜query｜>"),
            "task key must not survive; {text}"
        );
        assert!(text.contains("U1\n\nA2<｜Assistant｜></think>"), "{text}");
    }

    #[test]
    fn attach_task_errors_and_last_user_targeting() {
        assert_eq!(
            attach_task(&mut vec![json!({"role":"system","content":"s"})], "query"),
            Err(RenderErr::TaskWithoutUser)
        );
        assert_eq!(
            attach_task(&mut vec![json!({"role":"user","content":"u"})], "bogus"),
            Err(RenderErr::InvalidTask)
        );
        let mut msgs = vec![
            json!({"role":"user","content":"first"}),
            json!({"role":"assistant","content":"a"}),
            json!({"role":"user","content":"last"}),
        ];
        attach_task(&mut msgs, "query").unwrap();
        assert_eq!(msgs[0].get("task"), None);
        assert_eq!(msgs[2]["task"], json!("query"));
    }

    /// Non-`action` tasks append their special token WITHOUT the assistant
    /// opening; `action` opens an assistant turn first, with the think token
    /// keyed purely on the mode (no drop_thinking dependence).
    #[test]
    fn task_transition_tokens() {
        let user_task = |task: &str, thinking: bool| {
            render_request(
                &json!([{"role":"user","content":"do it"}]),
                None,
                RenderOpts {
                    thinking,
                    reasoning_effort: ReasoningEffort::None,
                },
                RequestParts {
                    task: Some(task),
                    continue_final_message: false,
                },
            )
            .unwrap()
            .0
        };
        assert_eq!(
            user_task("query", false),
            "<｜begin▁of▁sentence｜><｜User｜>do it<｜query｜>"
        );
        assert_eq!(
            user_task("action", false),
            "<｜begin▁of▁sentence｜><｜User｜>do it<｜Assistant｜></think><｜action｜>"
        );
        assert_eq!(
            user_task("action", true),
            "<｜begin▁of▁sentence｜><｜User｜>do it<｜Assistant｜><think><｜action｜>"
        );
        // message-level task key in the post-attach render layer (what the
        // engine's merge/render sees after `attach_task_to_last_user_message`
        // writes it onto a raw message).
        assert_eq!(
            render_messages(
                &json!([{"role":"user","content":"do it","task":"query"}]),
                None,
                RenderOpts::chat(),
            ),
            "<｜begin▁of▁sentence｜><｜User｜>do it<｜query｜>"
        );
    }

    /// Client-sent message-level `task` keys are stripped upstream (the
    /// engine's message model has no such field — only the pydantic-declared
    /// `request.task` path can ever produce a task transition). The strip
    /// applies to invalid names too: no engine assert can be triggered by a
    /// key the engine never sees.
    #[test]
    fn client_message_level_task_is_stripped() {
        let rendered = render_request(
            &json!([
                {"role":"user","content":"do it","task":"query"},
                {"role":"user","content":"do more","task":"bogus"}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts::default(),
        )
        .unwrap()
        .0;
        assert!(!rendered.contains("<｜query｜>"), "{rendered}");
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>do it\n\ndo more<｜Assistant｜></think>"
        );
        // roles normalize to lowercase the same way.
        let rendered = render_request(
            &json!([{"role":"User","content":"hi"}]),
            None,
            RenderOpts::chat(),
            RequestParts::default(),
        )
        .unwrap()
        .0;
        assert_eq!(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>"
        );
    }

    /// `prev_has_task`: the assistant turn directly after a task-carrying
    /// message is a task OUTPUT — its thinking is never rendered, even in
    /// thinking mode with tools (which normally KEEPS prior reasoning).
    #[test]
    fn prev_has_task_suppresses_thinking() {
        let thinking = |with_task: bool| {
            let mut msgs = vec![
                json!({"role":"system","content":"s"}),
                json!({"role":"user","content":"q"}),
            ];
            if with_task {
                msgs[1]["task"] = json!("query");
            }
            msgs.push(
                json!({"role":"assistant","content":"answer","reasoning_content":"my reasoning"}),
            );
            msgs.push(json!({"role":"user","content":"next"}));
            let tools = json!([{"type":"function","function":{"name":"f"}}]);
            render_messages(
                &serde_json::Value::Array(msgs),
                Some(&tools),
                RenderOpts {
                    thinking: true,
                    reasoning_effort: ReasoningEffort::None,
                },
            )
        };
        assert!(thinking(false).contains("my reasoning</think>"));
        assert!(
            !thinking(true).contains("my reasoning"),
            "task output must not render its reasoning"
        );
        // EOS/tool-calls still render around it.
        assert!(thinking(true).contains("answer<｜end▁of▁sentence｜>"));
    }

    /// Ordering: surgery first, then `task` attach (engine pipeline). With
    /// `continue_final_message` unset the trailing assistant is rewritten to
    /// user BEFORE the attach sees the list — and the preceding plain user
    /// run then MERGES it in, which drops the incoming task key exactly like
    /// `encoding_dsv4.merge_tool_messages` (its run-append guard only carries
    /// blocks). So this shape emits the plain generation prompt, not a task
    /// token — task-special-token rendering requires the task to land on a
    /// run-terminating message.
    #[test]
    fn surgery_runs_before_task_attach_and_merge_is_engine_faithful() {
        let rendered = render_request(
            &json!([
                {"role":"user","content":"U1"},
                {"role":"assistant","content":"A2"}
            ]),
            None,
            RenderOpts::chat(),
            RequestParts {
                task: Some("query"),
                continue_final_message: false,
            },
        )
        .unwrap()
        .0;
        assert_eq!(
            rendered, "<｜begin▁of▁sentence｜><｜User｜>U1\n\nA2<｜Assistant｜></think>",
            "task dropped by the user-run merge guard, as engine-side; {rendered}"
        );
    }

    /// Merge semantics with tasks: a task-carrying user turn TERMINATES a user
    /// run (a following user starts fresh), but a following TOOL message still
    /// folds INTO the task turn (the tool fold has no task guard engine-side).
    /// Transitions (including a task token) render only when the NEXT message
    /// is an assistant/reminder turn or the message is last — so a task on a
    /// user turn followed by another user turn emits no task token at all.
    #[test]
    fn task_merge_guard_and_tool_fold() {
        // {user(task), user}: no merge, and the first turn gets NO task
        // transition (next is a user, not an assistant turn).
        let text = render_messages(
            &json!([
                {"role":"user","content":"a","task":"query"},
                {"role":"user","content":"b"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>a<｜User｜>b<｜Assistant｜></think>"
        );

        // task on a NON-final user followed by an assistant turn: the task
        // token lands between the turns.
        let text = render_messages(
            &json!([
                {"role":"user","content":"a","task":"query"},
                {"role":"assistant","content":"b"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>a<｜query｜>b<｜end▁of▁sentence｜>"
        );

        // {user(task), tool}: the tool result folds into the task run's blocks.
        let text = render_messages(
            &json!([
                {"role":"user","content":"q","task":"query"},
                {"role":"tool","content":"result","tool_call_id":"c1"}
            ]),
            None,
            RenderOpts::chat(),
        );
        assert_eq!(
            text,
            "<｜begin▁of▁sentence｜><｜User｜>q\n\n<tool_result>result</tool_result><｜query｜>",
        );
    }
}
