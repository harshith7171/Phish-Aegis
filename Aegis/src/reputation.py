"""Opt-in third-party reputation checks for PhishGuard AI.

These functions do not fetch the submitted URL. When explicitly enabled they send it
to the configured reputation provider, which has its own privacy policy.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from typing import Any

import requests


@dataclass
class ReputationResult:
    provider: str
    status: str
    summary: str
    details: dict[str, Any]


def check_google_safe_browsing(url: str, api_key: str) -> ReputationResult:
    """Query Google Safe Browsing v4. The URL is sent to Google; target is not loaded."""
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = {
        "client": {"clientId": "phishguard-ai", "clientVersion": "2.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"], "threatEntryTypes": ["URL"], "threatEntries": [{"url": url}],
        },
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        matches = response.json().get("matches", [])
        if matches:
            types = sorted({match.get("threatType", "unknown") for match in matches})
            return ReputationResult("Google Safe Browsing", "flagged", "Known threat match: " + ", ".join(types), {"threat_types": types})
        return ReputationResult("Google Safe Browsing", "clear", "No match returned by this provider.", {})
    except requests.RequestException as exc:
        return ReputationResult("Google Safe Browsing", "unavailable", "Lookup failed; no conclusion was made.", {"error": str(exc)})


def check_virustotal(url: str, api_key: str) -> ReputationResult:
    """Look up an already-known URL at VirusTotal without submitting it for analysis."""
    url_id = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")
    try:
        response = requests.get(
            f"https://www.virustotal.com/api/v3/urls/{url_id}", headers={"x-apikey": api_key}, timeout=10,
        )
        if response.status_code == 404:
            return ReputationResult("VirusTotal", "unknown", "No existing VirusTotal record was found.", {})
        response.raise_for_status()
        stats = response.json()["data"]["attributes"].get("last_analysis_stats", {})
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        status = "flagged" if malicious or suspicious else "clear"
        summary = f"{malicious} malicious and {suspicious} suspicious engine verdict(s)."
        return ReputationResult("VirusTotal", status, summary, {"analysis_stats": stats})
    except requests.RequestException as exc:
        return ReputationResult("VirusTotal", "unavailable", "Lookup failed; no conclusion was made.", {"error": str(exc)})
    except (KeyError, TypeError, ValueError) as exc:
        return ReputationResult("VirusTotal", "unavailable", "Provider returned an unexpected response.", {"error": str(exc)})


def check_reputation(url: str, *, google_api_key: str | None = None, virustotal_api_key: str | None = None) -> list[dict[str, Any]]:
    """Run only provider checks with keys explicitly supplied by the user."""
    results: list[ReputationResult] = []
    if google_api_key:
        results.append(check_google_safe_browsing(url, google_api_key))
    if virustotal_api_key:
        results.append(check_virustotal(url, virustotal_api_key))
    return [asdict(result) for result in results]
