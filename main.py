import asyncio
import time
import logging
from router import Triage, Router, RouterConfig
from telemetry import Telemetry, RequestMetrics
import litellm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-router.main")

async def handle_request(prompt: str, triage: Triage, router: Router, telemetry: Telemetry):
    """
    Handles a single request through the production routing pipeline.
    """
    start_time = time.perf_counter()

    # 1. Triage Phase (System One Decision)
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
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content
        status = "success"
        error_msg = None
    except Exception as e:
        logger.error(f"Execution error with {selected_model}: {e}")
        result_text = f"Error: System failed to generate response using {selected_model}."
        status = "error"
        error_msg = str(e)

    exec_end = time.perf_counter()
    exec_latency = exec_end - exec_start

    # 4. Telemetry Recording
    metrics = RequestMetrics(
        prompt=prompt,
        triage_latency=triage_latency,
        execution_latency=exec_latency,
        selected_model=selected_model,
        status=status,
        error_message=error_msg
    )
    telemetry.record_request(metrics)

    return {
        "prompt": prompt,
        "score": score,
        "model": selected_model,
        "response": result_text
    }

async def main():
    # Initialize Production Components
    config = RouterConfig()
    triage = Triage(config)
    router = Router(config)
    telemetry = Telemetry()

    test_prompts = [
        "Hello! What is your name?",                                    # Expected: Low
        "Write a Python function to calculate Fibonacci numbers recursively.", # Expected: Medium
        "Compare the economic impact of the Industrial Revolution vs the Digital Revolution in terms of labor markets.", # Expected: High
        "What is 2+2?",                                                 # Expected: Low
    ]

    logger.info("Starting Production Router Tests...")

    # Run requests concurrently using asyncio.gather
    tasks = [handle_request(p, triage, router, telemetry) for p in test_prompts]
    results = await asyncio.gather(*tasks)

    # Print results summary
    print("\n" + "="*60)
    print(f"{'PROMPT':<40} | {'SCORE':<6} | {'MODEL':<15}")
    print("-" * 60)
    for res in results:
        print(f"{res['prompt'][:38]:<40} | {res['score']:<6} | {res['model']:<15}")
    print("="*60)

    # Final Performance Report
    print(telemetry.get_report())

if __name__ == "__main__":
    asyncio.run(main())
