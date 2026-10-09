import logging
from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import YamlConfigSettingsSource
import litellm

# Setup Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm-router")

class TriageResult(BaseModel):
    """Strict validation for the triage output."""
    complexity_score: int = Field(..., ge=0, le=10)
    reasoning: str

class RouterConfig(BaseSettings):
    """
    Production Configuration. Resolution order (highest to lowest priority):
    explicit constructor kwargs > environment variables > .env file > config.yaml > field defaults.
    """
    model_config = SettingsConfigDict(env_file=".env", yaml_file="config.yaml", extra="ignore", populate_by_name=True)

    triage_model: str = Field(default="gpt-4o-mini", alias="TRIAGE_MODEL")
    low_threshold: int = Field(default=4, alias="TRIAGE_LOW_THRESHOLD")
    high_threshold: int = Field(default=8, alias="TRIAGE_HIGH_THRESHOLD")
    timeout: int = Field(default=30, alias="TRIAGE_TIMEOUT")
    retry_attempts: int = Field(default=3, alias="RETRY_ATTEMPTS")
    max_prompt_length: int = Field(default=10_000, alias="MAX_PROMPT_LENGTH")
    model_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "low": "claude-haiku-4-5-20251001",
        "medium": "gpt-4o-mini",
        "high": "claude-sonnet-4-6",
    })

    # Per-tier env/.env overrides for individual model_mapping entries, merged in by _merge_model_overrides below.
    model_low: Optional[str] = Field(default=None, alias="MODEL_LOW")
    model_medium: Optional[str] = Field(default=None, alias="MODEL_MEDIUM")
    model_high: Optional[str] = Field(default=None, alias="MODEL_HIGH")

    @model_validator(mode="after")
    def _merge_model_overrides(self):
        for tier, override in (("low", self.model_low), ("medium", self.model_medium), ("high", self.model_high)):
            if override:
                self.model_mapping[tier] = override
        return self

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

class Triage:
    """
    Triage mimics Jev's 'System One' calibrated decisions.
    """
    def __init__(self, config: RouterConfig):
        self.config = config

    async def get_complexity(self, prompt: str) -> TriageResult:
        system_prompt = (
            "You are a calibrated decision engine. Your task is to score the complexity of the following user request. "
            "Score 0-3: Simple factual, greeting, or basic tasks. "
            "Score 4-7: Multi-step reasoning, synthesis, or moderate coding tasks. "
            "Score 8-10: Complex architecture, deep research, or high-stakes coding. "
            "Return ONLY a JSON object with 'complexity_score' (int) and 'reasoning' (short string)."
        )

        try:
            response = await litellm.acompletion(
                model=self.config.triage_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                timeout=self.config.timeout,
                num_retries=self.config.retry_attempts
            )

            content = response.choices[0].message.content
            return TriageResult.model_validate_json(content)

        except Exception as e:
            logger.error(f"Triage Critical Error: {e}")
            return TriageResult(complexity_score=5, reasoning="Fallback due to system error")

class Router:
    """
    Router maps a validated complexity score to a target model.
    """
    def __init__(self, config: RouterConfig):
        self.config = config

    def route(self, complexity_score: int) -> str:
        if complexity_score < self.config.low_threshold:
            return self.config.model_mapping["low"]
        elif complexity_score < self.config.high_threshold:
            return self.config.model_mapping["medium"]
        else:
            return self.config.model_mapping["high"]
