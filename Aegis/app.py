"""PhishGuard AI: URL, email, and SMS phishing-risk screening."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLUMNS, normalize_url
from src.reputation import check_reputation
from src.text_features import extract_urls, message_reasons, validate_text
from src.train import predict_bundle
from src.train_messages import predict_message_bundle

BASE_DIR = Path(__file__).resolve().parent

URL_MODEL_PATH = BASE_DIR / "models" / "phishguard_rf.joblib"
URL_METRICS_PATH = BASE_DIR / "models" / "metrics.json"

MESSAGE_MODEL_PATH = BASE_DIR / "models" / "phishguard_message_rf.joblib"
MESSAGE_METRICS_PATH = BASE_DIR / "models" / "message_metrics.json"
MAX_BATCH_ROWS = 1_000
MAX_LIVE_URLS = 5


@st.cache_resource
def load_url_bundle() -> dict:
    if not URL_MODEL_PATH.exists():
        raise FileNotFoundError("URL model not found. Run the URL training command in the README first.")
    bundle = joblib.load(URL_MODEL_PATH)
    if bundle.get("version") != 2:
        raise ValueError("The saved URL model is outdated. Run the URL training command in the README again.")
    return bundle


@st.cache_resource
def load_message_bundle() -> dict:
    if not MESSAGE_MODEL_PATH.exists():
        raise FileNotFoundError("Message model not found. Run `python -m src.train_messages` first.")
    bundle = joblib.load(MESSAGE_MODEL_PATH)
    if bundle.get("version") != 1:
        raise ValueError("The saved message model is outdated. Run `python -m src.train_messages` again.")
    return bundle


def score_label(score: float) -> tuple[str, str]:
    if score >= 0.80: return "Critical phishing risk", "error"
    if score >= 0.50: return "High phishing risk", "error"
    if score >= 0.25: return "Moderate phishing risk", "warning"
    return "Lower phishing risk", "success"


def local_url_reasons(features: dict[str, float]) -> list[str]:
    mapping = [
        ("has_ip_host", "the hostname is an IP address"), ("has_at_symbol", "it contains an @ symbol"),
        ("has_punycode", "the hostname uses punycode"), ("has_mixed_script", "the hostname mixes writing systems"),
        ("has_non_ascii", "it contains non-ASCII characters"), ("has_shortener", "it uses a URL-shortening service"),
        ("has_suspicious_tld", "the top-level domain needs extra scrutiny"), ("brand_in_subdomain", "a brand name appears in a subdomain"),
        ("has_double_dot", "it contains a double-dot sequence"), ("has_encoded_chars", "it contains encoded characters"),
    ]
    reasons = [message for key, message in mapping if features[key]]
    if features["suspicious_token_count"]: reasons.append("it contains account, login, security, or urgency-related terms")
    if features["subdomain_count"] >= 2: reasons.append("it has multiple subdomains")
    if features["digit_ratio"] >= 0.20: reasons.append("an unusually large share of the URL is numeric")
    if features["url_length"] >= 100: reasons.append("the URL is unusually long")
    return reasons or ["no single strong local warning pattern was detected"]


def secret_value(name: str) -> str:
    try:
        return os.getenv(name, "") or str(st.secrets.get(name, ""))
    except (FileNotFoundError, AttributeError):
        return os.getenv(name, "")


def display_reputation(results: list[dict]) -> None:
    if not results:
        st.info("No live provider key is configured. Local analysis is complete and no URL was shared externally.")
        return
    for result in results:
        message = f"{result['provider']}: {result['summary']}"
        if result["status"] == "flagged": st.error(message)
        elif result["status"] == "clear": st.success(message)
        elif result["status"] == "unknown": st.info(message)
        else: st.warning(message)


def live_checks(urls: list[str]) -> None:
    if len(urls) > MAX_LIVE_URLS:
        st.warning(f"Live checks are limited to the first {MAX_LIVE_URLS} links in a message.")
    for url in urls[:MAX_LIVE_URLS]:
        with st.expander(f"Reputation: {url}"):
            display_reputation(check_reputation(
                url, google_api_key=secret_value("GOOGLE_SAFE_BROWSING_API_KEY"),
                virustotal_api_key=secret_value("VIRUSTOTAL_API_KEY"),
            ))


st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ PhishGuard AI")
st.caption("Layered phishing-risk screening for URLs, email, and SMS. The app never opens submitted links or attachments.")
try:
    url_bundle = load_url_bundle()
    message_bundle = load_message_bundle()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(str(error))
    st.stop()

url_tab, message_tab, batch_tab, methodology_tab = st.tabs(["Check URL", "Check email or SMS", "Batch check", "Safety & methodology"])
with url_tab:
    submitted = st.text_input("Website URL", placeholder="https://example.com/login")
    allow_reputation = st.checkbox("Also check third-party reputation services", value=False, key="url_reputation", help="If enabled, the full URL is shared only with providers whose API keys you configure. The website itself is not opened.")
    if st.button("Assess URL risk", type="primary"):
        try:
            normalized = normalize_url(submitted)
            probabilities, feature_table = predict_bundle(url_bundle, [normalized])
            score = float(probabilities[0])
            label, status = score_label(score)
            getattr(st, status)(f"{label} — local model score: {score:.1%}")
            st.progress(score, text="Local phishing-risk score")
            st.warning("This is a risk signal, not a safety guarantee. Do not enter credentials, OTPs, or payment details based on this result alone.")
            left, right = st.columns([3, 2])
            with left:
                st.subheader("Observed local signals")
                for reason in local_url_reasons(feature_table.iloc[0].to_dict()): st.write(f"• {reason}")
            with right:
                st.subheader("Live reputation (optional)")
                if allow_reputation: live_checks([normalized])
                else: st.info("Disabled — your submitted URL remains local to this app.")
            with st.expander("Inspect local URL features"):
                st.dataframe(feature_table.T.rename(columns={0: "value"}), width="stretch")
        except ValueError as error:
            st.error(str(error))

with message_tab:
    channel = st.radio("Message type", ["Email", "SMS"], horizontal=True)
    pasted_message = st.text_area(f"Paste the {channel} text", height=220, placeholder="Paste the message body here. Links and attachments will not be opened.")
    allow_message_reputation = st.checkbox("Check visible links with third-party reputation services", value=False, key="message_reputation", help="If enabled, each visible URL is shared with configured providers. The linked website is not opened.")
    if st.button("Assess message risk", type="primary"):
        try:
            message = validate_text(pasted_message)
            message_probabilities, message_features = predict_message_bundle(message_bundle, [message])
            message_score = float(message_probabilities[0])
            visible_urls = [normalize_url(url) for url in extract_urls(message)]
            url_scores, _ = predict_bundle(url_bundle, visible_urls) if visible_urls else ([], None)
            strongest_url_score = float(max(url_scores)) if len(url_scores) else 0.0
            conservative_score = max(message_score, strongest_url_score)
            label, status = score_label(conservative_score)
            getattr(st, status)(f"{label} — conservative combined score: {conservative_score:.1%}")
            st.caption(f"{channel} text-model score: {message_score:.1%}" + (f" · highest visible-link score: {strongest_url_score:.1%}" if visible_urls else " · no visible links found"))
            st.warning("Do not reply, click links, open attachments, share credentials, or provide OTPs until you independently verify an unexpected message.")
            left, right = st.columns([3, 2])
            with left:
                st.subheader("Observed message signals")
                for reason in message_reasons(message_features.iloc[0].to_dict()): st.write(f"• {reason}")
            with right:
                st.subheader("Visible links")
                if visible_urls:
                    link_results = pd.DataFrame({"url": visible_urls, "phishing_risk_score": [round(float(score), 4) for score in url_scores], "risk_level": [score_label(float(score))[0] for score in url_scores]})
                    st.dataframe(link_results, width="stretch", hide_index=True)
                else: st.info("No visible http/https links were found in this message.")
            if allow_message_reputation:
                st.subheader("Live reputation (optional)")
                if visible_urls: live_checks(visible_urls)
                else: st.info("No visible links are available for a reputation lookup.")
            with st.expander("Inspect local message features"):
                st.dataframe(message_features.T.rename(columns={0: "value"}), width="stretch")
        except ValueError as error:
            st.error(str(error))

with batch_tab:
    st.caption("Upload a CSV with either a `url` column or a `text` column. Batch scoring is local only; nothing is sent to reputation providers.")
    uploaded = st.file_uploader("URL or message CSV", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            if len(batch) > MAX_BATCH_ROWS: raise ValueError(f"Limit batch uploads to {MAX_BATCH_ROWS:,} rows.")
            result = batch.copy()
            if "url" in batch.columns:
                normalized = [normalize_url(value) for value in batch.url.tolist()]
                scores, _ = predict_bundle(url_bundle, normalized)
                result["normalized_url"] = normalized
                result["phishing_risk_score"] = scores.round(4)
            elif "text" in batch.columns:
                messages = [validate_text(value) for value in batch.text.tolist()]
                scores, _ = predict_message_bundle(message_bundle, messages)
                result["phishing_risk_score"] = scores.round(4)
            else:
                raise ValueError("The CSV must include either a url column or a text column.")
            result["risk_level"] = [score_label(float(score))[0] for score in scores]
            st.dataframe(result, width="stretch")
            st.download_button("Download results CSV", result.to_csv(index=False).encode("utf-8"), "phishguard_results.csv", "text/csv")
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))

with methodology_tab:
    st.subheader("What is phishing?")
    st.write("Phishing is a cyberattack that uses deceptive emails, messages, calls, or websites to trick people into sharing sensitive information, installing malware, or taking another unsafe action.")
    st.info("A phishing-risk score is one signal. Pause and independently verify unexpected requests for passwords, OTPs, payment details, attachments, or downloads.")
    st.subheader("How the models work")
    st.write("The URL model combines static URL features and character n-grams. The email/SMS model combines local message features—such as urgency, credential requests, payment requests, threats, attachment language, and visible links—with character n-grams. Neither model opens a submitted link or attachment.")
    st.write("The live option queries Google Safe Browsing and/or VirusTotal only if you configure their API keys and explicitly turn it on. Each provider receives the submitted URL under its own privacy terms.")
    metric_columns = st.columns(2)
    for column, title, path, keys in [
        (metric_columns[0], "URL model evaluation", URL_METRICS_PATH, ["accuracy", "precision_phishing", "recall_phishing", "f1_phishing", "roc_auc", "average_precision", "train_rows", "test_rows", "domain_overlap", "feature_count"]),
        (metric_columns[1], "Email/SMS model evaluation", MESSAGE_METRICS_PATH, ["accuracy", "precision_phishing", "recall_phishing", "f1_phishing", "roc_auc", "average_precision", "train_rows", "test_rows", "source_group_overlap", "feature_count"]),
    ]:
        with column:
            st.subheader(title)
            if path.exists():
                metrics = json.loads(path.read_text(encoding="utf-8"))
                st.caption(metrics["evaluation"] + ". Metrics are from the most recent local training run.")
                st.json({key: metrics[key] for key in keys})
            else:
                st.info("Train this model to display measured evaluation results.")
