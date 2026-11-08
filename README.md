# Pakistan Cyber Law RAG — Legal-Information Retrieval Prototype

A Streamlit application that retrieves relevant passages from a supplied Pakistan cyber-law PDF and can use an xAI Grok model to produce a source-grounded explanation. The interface also supports a retrieval-only mode so users can inspect the passages found by the search step without calling an LLM.

**Live prototype:** https://pakistan-cyber-law-rag-application-2yvgz28qknksrwmvrvd6hs.streamlit.app/

> **Important:** this is an educational legal-information retrieval prototype, not legal advice, legal representation or an authoritative legal database. The source PDF may not reflect the latest amendments or Gazette notifications. Verify current law against official sources and consult a qualified Pakistani lawyer for a real matter.

## What it does

1. Downloads the configured PDF from Google Drive.
2. Extracts text page by page with page metadata.
3. Splits page text into overlapping word-based chunks.
4. Generates normalized embeddings with Sentence Transformers.
5. Builds a FAISS inner-product index.
6. Embeds a user's question and retrieves the most similar passages.
7. Optionally sends the retrieved context to the xAI API for an answer.
8. Displays the source excerpts and PDF page numbers for review.

The application has a retrieval-only fallback when an LLM response is unavailable or disabled.

## Architecture

~~~text
Google Drive PDF
       |
       v
gdown download
       |
       v
pypdf page extraction
       |
       v
Page-aware word chunking
(230 words, 45-word overlap)
       |
       v
all-MiniLM-L6-v2 embeddings
(normalized)
       |
       v
FAISS IndexFlatIP
       |
       v
Question embedding + top-K retrieval
       |
       +-------------------------+
       |                         |
       v                         v
Retrieved passages         Retrieval-only display
       |
       v
xAI OpenAI-compatible API (optional)
       |
       v
Grounded explanation with PDF page references
~~~

## Technology stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF retrieval source | Google Drive shared PDF |
| Download | gdown |
| Text extraction | pypdf |
| Chunking | Page-aware word chunks with overlap |
| Embeddings | Sentence Transformers, all-MiniLM-L6-v2 |
| Vector index | FAISS IndexFlatIP |
| LLM API | xAI OpenAI-compatible endpoint |
| Client library | openai Python package |
| Numeric operations | NumPy |

The application code currently defines the xAI API base URL as https://api.x.ai/v1 and a configurable Grok model name.

## Repository structure

~~~text
PAKISTAN-CYBER-LAW-RAG-APPLICATION/
├── app.py
├── requirements.txt
├── LICENSE
└── readme.md
~~~

The application creates a local .rag_data directory at runtime for the downloaded PDF and generated retrieval artifacts. These are runtime files, not source documents to commit.

## Run locally

### 1. Clone and install

~~~bash
git clone https://github.com/Qaisar-baloch/PAKISTAN-CYBER-LAW-RAG-APPLICATION.git
cd PAKISTAN-CYBER-LAW-RAG-APPLICATION
python -m venv .venv
~~~

Activate the environment:

**Windows**

~~~powershell
.venv\Scripts\activate
~~~

**macOS / Linux**

~~~bash
source .venv/bin/activate
~~~

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

### 2. Configure xAI credentials

Create .streamlit/secrets.toml:

~~~toml
XAI_API_KEY = "your_xai_api_key"
GROK_MODEL = "your_available_grok_model"
~~~

The app also reads XAI_API_KEY and GROK_MODEL environment variables. Never commit real credentials.

### 3. Launch

~~~bash
streamlit run app.py
~~~

## Source document access

The default source is a Google Drive PDF configured in app.py. The file must be accessible to the deployed runtime. The sidebar allows the source URL to be changed and provides a control to re-download the PDF and rebuild the index.

Only use documents you are permitted to access and process. Avoid uploading private case files or confidential personal information to a public deployment.

## Retrieval controls

The sidebar provides controls for:

- xAI API key and model name
- Whether to use Grok for the final answer
- Response technicality and length
- Temperature
- Top-K retrieval count
- Similarity threshold
- Visibility of retrieved passages
- Google Drive PDF URL
- Re-download and rebuild
- Strict legal-source mode

## Responsible legal use

The answer prompt instructs the model to:

- rely only on retrieved passages from the supplied PDF
- avoid inventing sections, penalties, authorities, dates or procedures
- state when the retrieved source does not support an answer
- cite material claims using PDF page numbers
- distinguish the text of the source from interpretation
- recommend qualified Pakistani legal counsel for case-specific questions

These prompt instructions reduce unsupported answers but cannot guarantee legal correctness. Retrieval can miss relevant passages, the PDF can be outdated, and a language model can still make mistakes.

## Limitations

- The source corpus is a supplied PDF, not a continuously updated official legal database.
- The application requires access to the Google Drive file during initial loading.
- Scanned PDFs without an extractable text layer may require OCR before indexing.
- FAISS indexes and downloaded files are stored in the application's local runtime directory.
- Hosting platforms with ephemeral storage may discard generated files after restart or redeployment.
- The application is a legal-information experiment, not a decision-support system for real cases.

## License

See [LICENSE](LICENSE) for the repository license.
