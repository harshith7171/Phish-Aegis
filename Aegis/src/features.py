"""Offline URL-string feature extraction for PhishGuard AI.

This module never makes a network request, resolves a domain, or loads a URL.
"""
from __future__ import annotations

import ipaddress
import math
import re
import unicodedata
from collections import Counter
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

COMMON_MULTI_LABEL_SUFFIXES = frozenset({
    "ac.in", "ac.jp", "ac.uk", "co.in", "co.jp", "co.nz", "co.uk", "co.za", "com.au", "com.br",
    "com.cn", "com.mx", "com.sg", "com.tr", "edu.au", "gov.au", "gov.in", "gov.uk", "net.au", "org.au",
    "org.uk",
})
FEATURE_COLUMNS = [
    "url_length", "hostname_length", "path_length", "query_length", "dot_count",
    "hyphen_count", "underscore_count", "digit_count", "digit_ratio", "special_char_count",
    "subdomain_count", "path_depth", "query_parameter_count", "has_https", "has_ip_host",
    "has_at_symbol", "has_punycode", "has_port", "has_encoded_chars", "has_fragment",
    "has_double_slash_path", "has_double_dot", "has_shortener", "has_suspicious_tld",
    "has_non_ascii", "has_mixed_script", "suspicious_token_count", "brand_token_count",
    "brand_in_subdomain", "entropy",
]
SUSPICIOUS_TOKENS = frozenset({
    "login", "signin", "verify", "verification", "secure", "security", "account", "update",
    "confirm", "password", "wallet", "banking", "payment", "invoice", "gift", "bonus",
    "urgent", "suspended", "recover", "unlock", "support", "authenticate",
})
BRAND_TOKENS = frozenset({
    "apple", "amazon", "bankofamerica", "chase", "coinbase", "docusign", "dropbox", "facebook",
    "google", "instagram", "microsoft", "netflix", "office", "outlook", "paypal", "telegram",
    "whatsapp", "yahoo",
})
URL_SHORTENERS = frozenset({
    "bit.ly", "buff.ly", "cutt.ly", "goo.gl", "is.gd", "lnkd.in", "ow.ly", "rb.gy", "rebrand.ly",
    "shorturl.at", "t.co", "tinyurl.com", "urlz.fr",
})
SUSPICIOUS_TLDS = frozenset({"app", "biz", "click", "country", "gq", "info", "live", "monster", "rest", "review", "ru", "shop", "tk", "top", "vip", "work", "xyz"})


def normalize_url(value: object) -> str:
    """Validate a URL for static parsing only; this does not fetch the URL."""
    if not isinstance(value, str):
        raise ValueError("Enter a URL as text.")
    url = value.strip()
    if not url:
        raise ValueError("Enter a URL.")
    if len(url) > 2_048:
        raise ValueError("URL is too long (maximum 2,048 characters).")
    if any(ord(char) < 32 or char.isspace() for char in url):
        raise ValueError("URLs cannot contain spaces or control characters.")
    if "://" not in url:
        url = "http://" + url
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported.")
    if not parsed.hostname:
        raise ValueError("Enter a URL with a hostname, for example example.com.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc
    return url


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def registered_domain(value: object) -> str:
    """Return a conservative domain grouping key; never contacts DNS or a suffix service."""
    try:
        host = (urlsplit(normalize_url(value)).hostname or "").lower().rstrip(".")
    except ValueError:
        return "invalid"
    if _is_ip(host):
        return host
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return host or "invalid"
    suffix_length = 2 if ".".join(labels[-2:]) in COMMON_MULTI_LABEL_SUFFIXES else 1
    return ".".join(labels[-(suffix_length + 1):]) if len(labels) > suffix_length else host


def _contains_mixed_script(text: str) -> bool:
    scripts: set[str] = set()
    for char in text:
        if char.isascii() or not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "HIRAGANA", "KATAKANA", "HANGUL", "CJK"):
            if script in name:
                scripts.add(script)
                break
    return bool(scripts and any(char.isascii() and char.isalpha() for char in text)) or len(scripts) > 1


def extract_features(value: object) -> dict[str, float]:
    """Produce deterministic numerical features from a URL string without opening it."""
    url = normalize_url(value)
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    registered = registered_domain(url)
    path_query = f"{parsed.path}?{parsed.query}".lower()
    words = re.findall(r"[a-z]+", f"{host} {path_query}")
    counts = Counter(url.lower())
    entropy = -sum((count / len(url)) * math.log2(count / len(url)) for count in counts.values())
    labels = [part for part in host.split(".") if part]
    suffix = host.rsplit(".", 1)[-1] if not _is_ip(host) else ""
    subdomain = host[: -(len(registered) + 1)] if registered and host.endswith("." + registered) else ""
    alpha_numeric = sum(char.isalnum() for char in url)
    special = sum(char in "@?&=%_~" for char in url)
    return {
        "url_length": len(url), "hostname_length": len(host), "path_length": len(parsed.path),
        "query_length": len(parsed.query), "dot_count": url.count("."), "hyphen_count": url.count("-"),
        "underscore_count": url.count("_"), "digit_count": sum(char.isdigit() for char in url),
        "digit_ratio": sum(char.isdigit() for char in url) / max(1, alpha_numeric), "special_char_count": special,
        "subdomain_count": max(0, len(labels) - len(registered.split("."))),
        "path_depth": len([part for part in parsed.path.split("/") if part]),
        "query_parameter_count": 0 if not parsed.query else len(parsed.query.split("&")),
        "has_https": float(parsed.scheme.lower() == "https"), "has_ip_host": float(_is_ip(host)),
        "has_at_symbol": float("@" in url), "has_punycode": float("xn--" in host),
        "has_port": float(parsed.port is not None), "has_encoded_chars": float("%" in url),
        "has_fragment": float(bool(parsed.fragment)), "has_double_slash_path": float("//" in parsed.path),
        "has_double_dot": float(".." in url), "has_shortener": float(host in URL_SHORTENERS),
        "has_suspicious_tld": float(suffix in SUSPICIOUS_TLDS), "has_non_ascii": float(not url.isascii()),
        "has_mixed_script": float(_contains_mixed_script(host)),
        "suspicious_token_count": float(sum(word in SUSPICIOUS_TOKENS for word in words)),
        "brand_token_count": float(sum(word in BRAND_TOKENS for word in words)),
        "brand_in_subdomain": float(any(brand in subdomain for brand in BRAND_TOKENS)), "entropy": entropy,
    }


def feature_frame(urls: pd.Series | list[str]) -> pd.DataFrame:
    """Extract an ordered feature frame, failing clearly on malformed dataset rows."""
    rows: list[dict[str, Any]] = []
    for index, url in enumerate(urls):
        try:
            rows.append(extract_features(url))
        except ValueError as exc:
            raise ValueError(f"Invalid URL at row {index + 1}: {exc}") from exc
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
