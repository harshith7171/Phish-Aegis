"""Safe, local-only feature extraction for email and SMS phishing screening."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

TEXT_FEATURE_COLUMNS = [
    "text_length", "word_count", "uppercase_ratio", "digit_ratio", "exclamation_count", "question_count",
    "url_count", "email_address_count", "phone_number_count", "currency_count", "suspicious_token_count",
    "has_urgency", "has_credential_request", "has_payment_request", "has_threat", "has_prize",
    "has_attachment", "has_generic_greeting", "has_brand_name", "has_obfuscation",
]
SUSPICIOUS_TERMS = frozenset({
    "account", "authenticate", "bank", "bonus", "claim", "confirm", "credential", "gift", "immediately",
    "invoice", "login", "otp", "password", "payment", "refund", "secure", "signin", "suspended", "urgent",
    "validate", "verify", "wallet",
})
URGENCY_TERMS = frozenset({"act now", "asap", "immediately", "limited time", "urgent", "within 24", "today"})
CREDENTIAL_TERMS = frozenset({"password", "otp", "one-time code", "login", "sign in", "credential", "verify your account"})
PAYMENT_TERMS = frozenset({"payment", "bank account", "card details", "gift card", "wire transfer", "crypto", "wallet"})
THREAT_TERMS = frozenset({"suspended", "blocked", "legal action", "penalty", "deactivated", "compromised"})
PRIZE_TERMS = frozenset({"you won", "winner", "prize", "reward", "free gift", "bonus"})
ATTACHMENT_TERMS = frozenset({"attachment", "attached file", ".zip", ".exe", ".html", "enable macros"})
GENERIC_GREETINGS = frozenset({"dear customer", "dear user", "valued customer", "dear account holder"})
BRANDS = frozenset({"amazon", "apple", "bank of america", "chase", "coinbase", "docusign", "google", "microsoft", "netflix", "paypal", "whatsapp"})
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>()\[\]{}\"']+")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")


def validate_text(value: object) -> str:
    """Validate pasted email/SMS text without accessing anything it contains."""
    if not isinstance(value, str):
        raise ValueError("Enter message text.")
    text = value.strip()
    if not text:
        raise ValueError("Enter an email or SMS message.")
    if len(text) > 20_000:
        raise ValueError("Message is too long (maximum 20,000 characters).")
    if any(ord(char) < 9 or (13 < ord(char) < 32) for char in text):
        raise ValueError("Message contains unsupported control characters.")
    return text


def extract_urls(text: object) -> list[str]:
    """Extract visible URLs only; no redirect resolution or network activity occurs."""
    message = validate_text(text)
    urls = [match.rstrip(".,;:!?'") for match in URL_PATTERN.findall(message)]
    return list(dict.fromkeys(urls))


def _contains_term(text: str, terms: frozenset[str]) -> bool:
    return any(term in text for term in terms)


def extract_text_features(value: object) -> dict[str, float]:
    """Turn message text into safe lexical features for the message model."""
    text = validate_text(value)
    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)
    letters = sum(char.isalpha() for char in text)
    upper = sum(char.isupper() for char in text)
    alpha_numeric = sum(char.isalnum() for char in text)
    obfuscation = "\u200b" in text or "\ufeff" in text or bool(re.search(r"(?:[. ]|\*){2,}", text))
    return {
        "text_length": len(text), "word_count": len(words), "uppercase_ratio": upper / max(1, letters),
        "digit_ratio": sum(char.isdigit() for char in text) / max(1, alpha_numeric),
        "exclamation_count": text.count("!"), "question_count": text.count("?"), "url_count": len(extract_urls(text)),
        "email_address_count": len(EMAIL_PATTERN.findall(text)), "phone_number_count": len(PHONE_PATTERN.findall(text)),
        "currency_count": sum(text.count(symbol) for symbol in "$€£₹"),
        "suspicious_token_count": float(sum(word in SUSPICIOUS_TERMS for word in words)),
        "has_urgency": float(_contains_term(lowered, URGENCY_TERMS)),
        "has_credential_request": float(_contains_term(lowered, CREDENTIAL_TERMS)),
        "has_payment_request": float(_contains_term(lowered, PAYMENT_TERMS)),
        "has_threat": float(_contains_term(lowered, THREAT_TERMS)), "has_prize": float(_contains_term(lowered, PRIZE_TERMS)),
        "has_attachment": float(_contains_term(lowered, ATTACHMENT_TERMS)),
        "has_generic_greeting": float(_contains_term(lowered, GENERIC_GREETINGS)),
        "has_brand_name": float(_contains_term(lowered, BRANDS)), "has_obfuscation": float(obfuscation),
    }


def text_feature_frame(messages: list[str] | pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        try:
            rows.append(extract_text_features(message))
        except ValueError as exc:
            raise ValueError(f"Invalid message at row {index + 1}: {exc}") from exc
    return pd.DataFrame(rows, columns=TEXT_FEATURE_COLUMNS)


def message_reasons(features: dict[str, float]) -> list[str]:
    """Return transparent observed warning signs, independently of model probability."""
    mapping = [
        ("has_urgency", "uses urgency or a deadline"), ("has_credential_request", "requests credentials or a verification code"),
        ("has_payment_request", "asks for payment or financial details"), ("has_threat", "threatens a negative consequence"),
        ("has_prize", "offers a prize, gift, or bonus"), ("has_attachment", "mentions an attachment or potentially risky file"),
        ("has_generic_greeting", "uses a generic greeting"), ("has_obfuscation", "contains possible text obfuscation"),
    ]
    reasons = [description for feature, description in mapping if features[feature]]
    if features["url_count"]:
        reasons.append("contains one or more visible links")
    if features["suspicious_token_count"] >= 2:
        reasons.append("contains several account/security-related terms")
    return reasons or ["no single strong local message warning sign was detected"]
