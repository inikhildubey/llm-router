# Intelligent LLM Router Service
A routing system inspired by [Jev](https://typesafe.ai), TypeSafe AI's "System One" model that makes fast, calibrated decisions instead of generating free-form text. This service applies that idea to score the complexity of an incoming request and route it to the most cost-effective LLM.

## 🚀 Architecture
This service acts as an **Intelligent Gateway**. Instead of sending every prompt to a powerful (and expensive) model, it uses a triage phase to determine the necessary "brain power" required.

1. **Triage**: An async, LLM-based complexity scoring step inspired by the System One decision-making concept, rating prompt complexity (0-10).
2. **Router**: Maps the score to a specific model based on environmental configuration.
3. **Execution**: Dispatches the request to the target model via `litellm`.
4. **Telemetry**: Records latency and model distribution for observability.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd "An Intelligent LLM Router"
```

### 2. Set Up Environment
```bash
pip install -r requirements.txt
cp .env.example .env
```
Edit `.env` and add your API keys and preferred models.

### 3. Run the Service
```bash
uvicorn app:app --reload
```
The service will be available at `http://127.0.0.1:8000`.

## ✅ Testing
```bash
pytest
```
Covers router threshold logic, triage JSON parsing/fallback behavior (mocked), and the FastAPI endpoints (health, success, validation, execution failure, metrics) via `TestClient` — no real API keys or network calls required. CI runs this suite on every push/PR via [.github/workflows/ci.yml](.github/workflows/ci.yml).

## 📡 API Documentation

### Route Request
**Endpoint:** `POST /route`
**Payload:**
```json
{
  "prompt": "Your user prompt here"
}
```
**Response (200):**
```json
{
  "prompt": "...",
  "score": 7,
  "model": "gpt-4o-mini",
  "response": "...",
  "triage_latency": 0.45,
  "execution_latency": 1.20
}
```
**422** — `prompt` is empty/whitespace-only, or exceeds `MAX_PROMPT_LENGTH`.
**502** — the execution call failed after retries (triage failures instead fail open to the medium tier, so they never surface as an HTTP error).

### Health Check
**Endpoint:** `GET /health`

### Metrics Report
**Endpoint:** `GET /metrics`

## ⚙️ Configuration
Settings resolve in this order (highest to lowest priority): environment variables > `.env` file > `config.yaml` > built-in defaults.
- `TRIAGE_MODEL`: The model used for scoring.
- `TRIAGE_LOW_THRESHOLD` / `TRIAGE_HIGH_THRESHOLD`: Score thresholds that select the low/medium/high tier.
- `MODEL_LOW` / `MODEL_MEDIUM` / `MODEL_HIGH`: The specific model IDs for each tier.
- `RETRY_ATTEMPTS`: Number of retries litellm applies to both the triage and execution calls.
- `MAX_PROMPT_LENGTH`: Maximum accepted prompt length; longer prompts are rejected with HTTP 422.

If an execution call ultimately fails after retries, `/route` responds with HTTP 502 rather than a 200 with an error string in the body. Triage failures still fail open to the medium tier.

## 🐳 Running with Docker
```bash
docker build -t llm-router .
docker run --env-file .env -p 8000:8000 llm-router
```
