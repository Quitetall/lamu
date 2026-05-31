//! Council — multi-model blind vote + synthesis (PewDiePie's "Council"
//! pattern, ported). Runs the SAME prompt across N models concurrently,
//! then a judge model picks the best ANONYMIZED answer and synthesizes a
//! final one combining their strengths. Built on the same local+cloud
//! dispatch as `parallel_query`.

use crate::server::LamuMcpServer;
use serde_json::{json, Value};

const JUDGE_PROMPT: &str = "\
You are judging anonymized answers (labeled A, B, C, …) to the same \
question. Identify the single best answer, then synthesize a final answer \
that combines their strengths and corrects their errors. Reply ONLY as a \
JSON object: {\"best\": \"<letter>\", \"synthesis\": \"<final answer>\", \
\"reasoning\": \"<one short line>\"}.";

/// Parse the judge reply → (best_letter, synthesis). Tolerant of code
/// fences / surrounding prose. None on unparseable.
pub(crate) fn parse_judge_verdict(reply: &str) -> Option<(String, String)> {
    let t = reply
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let try_obj = |s: &str| -> Option<(String, String)> {
        let v: Value = serde_json::from_str(s).ok()?;
        let best = v.get("best")?.as_str()?.trim().to_string();
        let synth = v.get("synthesis")?.as_str()?.to_string();
        Some((best, synth))
    };
    if let Some(r) = try_obj(t) {
        return Some(r);
    }
    // Fallback: first {...} object in the reply.
    if let (Some(s), Some(e)) = (reply.find('{'), reply.rfind('}')) {
        if e > s {
            return try_obj(&reply[s..=e]);
        }
    }
    None
}

pub async fn handle_council(server: &LamuMcpServer, args: Value) -> String {
    let prompt = args["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return "error: council requires a non-empty `prompt`".into();
    }
    let system = args["system"].as_str().unwrap_or("").to_string();
    let judge = args["judge_model"].as_str().unwrap_or("mimo-v2.5-pro").to_string();
    let include_answers = args["include_answers"].as_bool().unwrap_or(true);
    let models: Vec<String> = args["models"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect::<Vec<_>>())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec!["mimo-v2.5".into(), "deepseek-v4-flash".into()]);
    if models.len() < 2 {
        return "error: council needs >= 2 models (pass `models: [..]`)".into();
    }

    // Routing gate: under local-only, refuse cloud members (mirror
    // parallel_query) — but here we just run what's allowed and let the
    // judge work over the survivors.
    let local_only = server.routing_mode.lock().await.as_str() == "local-only";
    let cloud = crate::cloud::load_cloud_models();

    // Fan out: same prompt to every model, concurrently.
    let futs = models.iter().cloned().map(|model| {
        let is_cloud = cloud.iter().any(|m| m.name == model);
        let inner = json!({
            "model": model,
            "prompt": prompt,
            "system": system,
            "max_tokens": 4096,
            "temperature": 0.3,
        });
        let refuse = is_cloud && local_only;
        async move {
            let ans = if refuse {
                format!("error: cloud model '{model}' refused — routing mode is local-only")
            } else if is_cloud {
                crate::cloud::handle_cloud_query(inner).await
            } else {
                server.handle_query(inner).await
            };
            (model, ans)
        }
    });
    let results: Vec<(String, String)> = futures_util::future::join_all(futs).await;

    let good: Vec<&(String, String)> = results
        .iter()
        .filter(|(_, a)| !a.starts_with("error"))
        .collect();
    if good.len() < 2 {
        return format!(
            "error: council needs >= 2 successful answers; got {} of {} ({})",
            good.len(),
            models.len(),
            results
                .iter()
                .map(|(m, a)| format!("{m}: {}", if a.starts_with("error") { "err" } else { "ok" }))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Blind labels A, B, C, … for the judge.
    let labels: Vec<char> = ('A'..='Z').take(good.len()).collect();
    let blind = good
        .iter()
        .zip(&labels)
        .map(|((_, a), l)| format!("Answer {l}:\n{a}"))
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");
    let verdict_raw = crate::cloud::handle_cloud_query(json!({
        "model": judge,
        "system": JUDGE_PROMPT,
        "prompt": format!("Question:\n{prompt}\n\n{blind}"),
        "max_tokens": 4096,
        "temperature": 0.2,
        "include_reasoning": false,
    }))
    .await;

    let mut out = String::new();
    out.push_str(&format!("Council of {} (judge: {judge})\n", good.len()));
    for ((model, _), l) in good.iter().zip(&labels) {
        out.push_str(&format!("  [{l}] {model}\n"));
    }
    out.push('\n');

    match parse_judge_verdict(&verdict_raw) {
        Some((best, synth)) => {
            let winner = labels
                .iter()
                .position(|c| best.contains(*c))
                .map(|i| good[i].0.clone())
                .unwrap_or_else(|| best.clone());
            out.push_str(&format!("Winner: {winner} [{best}]\n\nSynthesis:\n{synth}\n"));
        }
        None => out.push_str(&format!("Judge reply (unparsed):\n{verdict_raw}\n")),
    }

    if include_answers {
        out.push_str("\n── individual answers ──\n");
        for ((model, ans), l) in good.iter().zip(&labels) {
            out.push_str(&format!("\n[{l}] {model}:\n{ans}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_judge_verdict_shapes() {
        let (b, s) = parse_judge_verdict(r#"{"best":"B","synthesis":"final","reasoning":"x"}"#).unwrap();
        assert_eq!(b, "B");
        assert_eq!(s, "final");
        let (b, s) =
            parse_judge_verdict("```json\n{\"best\": \"A\", \"synthesis\": \"hi\"}\n```").unwrap();
        assert_eq!(b, "A");
        assert_eq!(s, "hi");
        // prose-wrapped
        let (b, _) =
            parse_judge_verdict("After review: {\"best\":\"C\",\"synthesis\":\"z\"} done").unwrap();
        assert_eq!(b, "C");
        assert!(parse_judge_verdict("no json here").is_none());
    }
}
