# Pakistan Cyber Law RAG — Streamlit + FAISS + Grok

A production-oriented starter RAG application for questions about Pakistan cyber law.

It uses:

- **Streamlit** for the UI
- **FAISS** for vector similarity search
- **sentence-transformers** for local embeddings
- **Grok/xAI API** for grounded answer generation
- **pypdf** for extracting text from the legal PDF
- **gdown** to download the supplied Google Drive PDF automatically at startup

## Files

```text
app.py
requirements.txt
readme.md
```

## What the app does

1. Downloads the configured Google Drive PDF on startup.
2. Extracts the PDF text.
3. Splits it into overlapping chunks while retaining PDF page numbers.
4. Creates normalized sentence embeddings.
5. Builds a FAISS inner-product index.
6. Retrieves the most relevant legal passages for each question.
7. Optionally sends only those retrieved passages to Grok.
8. Displays the retrieved evidence so the user can inspect the legal source.

The default Drive file ID is the one supplied with the project:

```text
1znmUdzzNS5gFmbSa5zEV9QFYO--PGzbr
```

## Important: Google Drive permissions

The PDF must be accessible to the environment running the app.

Set the Google Drive file sharing to:

**Anyone with the link → Viewer**

If the file is private, the startup download will fail.

You can replace the Google Drive URL in the sidebar without changing the code.

## Run locally

Use Python 3.10+.

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS/Colab

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Run on Google Colab

In a Colab cell:

```python
!git clone YOUR_GITHUB_REPO_URL
%cd YOUR_REPO_FOLDER
!pip install -r requirements.txt
!streamlit run app.py &>/content/streamlit.log &
```

To expose Streamlit from Colab, use your preferred tunnel method (for example,
a supported Cloudflare/Colab-compatible tunnel). Do not hard-code API keys in
the notebook.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Put `app.py` and `requirements.txt` in the repository root.
3. Deploy the repository on Streamlit Community Cloud.
4. Select `app.py` as the entrypoint.
5. Add your xAI key in Streamlit Secrets.

Recommended secret:

```toml
XAI_API_KEY = "your_xai_api_key"
GROK_MODEL = "grok-4.6"
```

Do **not** commit API keys to GitHub.

The app also accepts `XAI_API_KEY` and `GROK_MODEL` environment variables.

## xAI / Grok

The app uses the OpenAI-compatible xAI endpoint:

```text
https://api.x.ai/v1
```

Default model:

```text
grok-4.6
```

The model is configurable in the sidebar, so it can be changed if your xAI
account exposes a different model.

## UI controls

The sidebar includes:

- xAI API key
- Grok model
- Use Grok toggle
- Technicality level
  - Beginner
  - Intermediate
  - Technical
  - Legal-professional
- Response size
  - Short
  - Medium
  - Detailed
- Temperature
- Top-K retrieval
- Similarity threshold
- Show retrieved passages
- Google Drive PDF URL
- Re-download + rebuild index
- Strict legal-source mode

## Legal-grounding design

The system prompt instructs Grok to:

- use only retrieved text from the supplied PDF
- avoid inventing sections, penalties, authorities, dates, or procedures
- say when the PDF does not provide enough evidence
- cite legal claims using PDF page numbers
- distinguish legal text from interpretation
- avoid operational instructions for harmful cyber activity
- recommend qualified Pakistani legal counsel for case-specific matters

### Why this matters

A general LLM can produce plausible but incorrect legal answers. This app
therefore makes the retrieved source passages visible and treats the supplied
PDF as the primary source.

## Current-law verification

The app should not be treated as an authoritative legal database.

The Pakistan Code website currently lists the Prevention of Electronic Crimes
Act, 2016, and its published PDF incorporates the **Prevention of Electronic
Crimes (Amendment) Act, 2025**. The Pakistan Code site also warns that its
content is for information purposes and may be under review.

For a real legal matter, verify the applicable law against the latest official
Gazette / Pakistan Code and consult a qualified Pakistani lawyer.

## Troubleshooting

### `Could not download the Google Drive PDF`

Check:

- Drive sharing is "Anyone with the link"
- the URL is a Google Drive file URL
- the file is actually a PDF
- the deployment environment has internet access

### `No text could be extracted`

The PDF may be scanned/image-only. OCR it first, then use the OCR PDF.

### Grok does not answer

Check:

- `XAI_API_KEY` is present
- the key has API access/credits
- the selected model is available to your xAI account

The app falls back to retrieval-only mode if Grok fails.

### FAISS installation issue

Use Python 3.10+ and deploy with the same Python major/minor version you use
for testing. If a platform has a newer/older Python runtime, choose a
compatible runtime or adjust the FAISS version.

## Security notes

- Never commit `XAI_API_KEY`.
- Use Streamlit Secrets or environment variables.
- Do not put private case documents into a public GitHub repository.
- This project is designed for legal-information retrieval, not automated
  legal decision-making.
