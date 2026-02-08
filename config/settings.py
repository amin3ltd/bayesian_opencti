from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    opencti_url: str = Field(..., alias="OPENCTI_URL")
    opencti_token: str = Field(..., alias="OPENCTI_TOKEN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    poll_interval_seconds: int = Field(30, alias="POLL_INTERVAL_SECONDS")
    max_parents_per_node: int = Field(5, alias="MAX_PARENTS_PER_NODE")
    model_config = SettingsConfigDict(env_file=".env.example", case_sensitive=True)

settings = Settings()
