# PhishGuard AI

PhishGuard AI is a layered phishing-risk screener for **URLs, email, and SMS** built with Python, scikit-learn Random Forest, and Streamlit. Its local analysis reads only pasted text: it does not resolve DNS, load a webpage, follow redirects, download content, or open attachments.

> A low score is not proof of safety; a high score is not proof of maliciousness. Never enter credentials, OTPs, or payment details because a tool marked a URL as safe.

## Capabilities

- **30 static URL features**: IP-host, URL shortener, port, `@`, encoding, Unicode/mixed-script and punycode checks, suspicious TLD, double-dot and double-slash patterns, path/query shape, digit ratio, entropy, brand-in-subdomain, and security/urgency terms.
- **Character n-gram learning**: detects suspicious URL shapes that hand-built rules miss.
- **Random Forest scoring**: combines the static features and character n-grams into a local phishing-risk probability.
- **Email and SMS screening**: analyzes local message signals including urgency, credential or OTP requests, payment requests, threats, prize language, attachment language, generic greetings, and visible links. Embedded links are separately screened by the URL model.
- **Leakage-aware evaluation**: removes exact duplicate URLs, rejects conflicting labels, and holds out whole domain-group keys using `StratifiedGroupKFold`.
- **Real-time reputation, by opt-in only**: supports Google Safe Browsing and VirusTotal. The submitted URL is sent to a provider only after the dashboard checkbox is enabled and a user-supplied key is configured. It never opens the target website.
- **Batch scoring**: upload a CSV with a `url` column and download local-only results.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.train --dataset data/sample_urls.csv
python -m src.train_messages --dataset data/sample_messages.csv
streamlit run app.py
```

Open the address Streamlit prints, usually `http://localhost:8501`.

## Optional live reputation checks

Create a private local settings file from the example:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Add your Google Safe Browsing and/or VirusTotal API key to `.streamlit/secrets.toml`, restart Streamlit, and enable **Also check third-party reputation services** in the dashboard. Do not commit `secrets.toml` or share the keys. The app queries providers; it does not visit the target URL.

Provider results mean only what that provider reported at lookup time. A “clear” or “unknown” response is not a safety guarantee.

## Use a real dataset

The included `data/sample_urls.csv` is a small offline demonstration dataset, not a benchmark or production training source. Acquire a current, licensed, URL-only dataset from sources you are allowed to use, such as a phishing-URL dataset on Kaggle or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). Review its license, label definitions, age, and sensitive-data risk. Do not crawl phishing pages to make a dataset.

Your CSV needs these columns:

```csv
url,label
https://legitimate-example.test/page,0
http://suspicious-example.test/login,1
```

`0` means legitimate and `1` means phishing. Train with:

```powershell
python -m src.train --dataset path\to\your_urls.csv --model models\phishguard_rf.joblib --metrics models\metrics.json
```

The training script reports measured accuracy, precision, recall, F1, ROC-AUC, average precision, a confusion matrix, and `domain_overlap`. `domain_overlap` must be `0`; the script fails rather than quietly allowing a leaky split. Domain keys conservatively handle common multi-part suffixes such as `co.uk`; use a maintained public-suffix library before deployment across all global domains.

### Email and SMS training data

`data/sample_messages.csv` is an intentionally small demonstration dataset. It has `text`, `label`, `channel`, and `source_group` columns. `text` and `label` are required; `channel` is descriptive; `source_group` lets the trainer keep messages from one campaign, sender, or source out of both training and test data.

```csv
text,label,channel,source_group
Your team meeting starts at 3 PM.,0,email,internal_meeting
URGENT: Verify your account at http://example.test/login,1,sms,credential_campaign
```

Train a real labeled message dataset with:

```powershell
python -m src.train_messages --dataset path\to\your_messages.csv --model models\phishguard_message_rf.joblib --metrics models\message_metrics.json
```

Use permitted, privacy-reviewed data only. Remove credentials, personal information, and sensitive identifiers before training. If `source_group` is absent, the script runs a row-wise split and labels that weaker evaluation clearly; add campaign/sender/source groups for a stronger held-out test.

## Recommended production path

For a security product, use this model as one signal in a layered service: fresh reputation feeds, an approved external sandbox, reviewed domain/DNS/certificate telemetry, audit logging with retention controls, rate limiting, authenticated API keys, monitoring for model drift, and human review for high-impact decisions. Never treat the model as a standalone security control.

## Project layout

```text
app.py                         Streamlit dashboard and batch workflow
src/features.py                Offline URL validation and advanced feature extraction
src/train.py                   Domain-disjoint ML training and evaluation
src/text_features.py           Offline email/SMS feature extraction and link detection
src/train_messages.py          Email/SMS ML training and evaluation
src/reputation.py              Opt-in provider lookups (never opens the target URL)
data/sample_urls.csv           Small offline demo dataset
data/sample_messages.csv       Small offline email/SMS demo dataset
models/                        Generated model and measured metrics
.streamlit/secrets.toml.example Optional local provider-key template
```
