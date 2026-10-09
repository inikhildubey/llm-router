import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from router import Triage, Router, RouterConfig
from telemetry import Telemetry, RequestMetrics
import litellm
import time
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-router-service")

app = FastAPI(
    title="Intelligent LLM Router Service",
    description="A Jev-inspired service that routes prompts to the most cost-effective model based on calibrated complexity scores.",
    version="1.0.0"
)

# Initialize Global Components
config = RouterConfig()
triage = Triage(config)
router = Router(config)
telemetry = Telemetry()

class RequestBody(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_meaningful(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("prompt must not be empty or whitespace-only")
        if len(stripped) > config.max_prompt_length:
            raise ValueError(f"prompt exceeds maximum length of {config.max_prompt_length} characters")
        return stripped

class ResponseBody(BaseModel):
    prompt: str
    score: int
    model: str
    response: str
    triage_latency: float
    execution_latency: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "triage_model": config.triage_model}

@app.post("/route", response_model=ResponseBody)
async def route_request(body: RequestBody):
    prompt = body.prompt
    start_time = time.perf_counter()

    # 1. Triage Phase
    triage_start = time.perf_counter()
    triage_result = await triage.get_complexity(prompt)
    triage_end = time.perf_counter()
    triage_latency = triage_end - triage_start

    score = triage_result.complexity_score

    # 2. Routing Phase
    selected_model = router.route(score)

    # 3. Execution Phase
    exec_start = time.perf_counter()
    try:
        response = await litellm.acompletion(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            timeout=config.timeout,
            num_retries=config.retry_attempts
        )
        result_text = response.choices[0].message.content
    except Exception as e:
        exec_latency = time.perf_counter() - exec_start
        logger.error(f"Execution error with {selected_model}: {e}")
        telemetry.record_request(RequestMetrics(
            prompt=prompt,
            triage_latency=triage_latency,
            execution_latency=exec_latency,
            selected_model=selected_model,
            status="error",
            error_message=str(e)
        ))
        raise HTTPException(status_code=502, detail=f"Execution failed for model {selected_model}: {e}")

    exec_end = time.perf_counter()
    exec_latency = exec_end - exec_start

    # 4. Telemetry
    metrics = RequestMetrics(
        prompt=prompt,
        triage_latency=triage_latency,
        execution_latency=exec_latency,
        selected_model=selected_model,
        status="success",
        error_message=None
    )
    telemetry.record_request(metrics)

    return ResponseBody(
        prompt=prompt,
        score=score,
        model=selected_model,
        response=result_text,
        triage_latency=triage_latency,
        execution_latency=exec_latency
    )

@app.get("/metrics")
async def get_metrics():
    return {"report": telemetry.get_report()}
