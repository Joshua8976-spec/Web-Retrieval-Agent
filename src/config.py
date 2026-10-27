from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings (OpenAI-compatible endpoint)
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="EMPTY", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")

    # Agent Loop Hard Constraints
    agent_max_pages: int = Field(default=10, alias="AGENT_MAX_PAGES")
    agent_timeout_seconds: int = Field(default=120, alias="AGENT_TIMEOUT_SECONDS")
    agent_max_retries: int = Field(default=3, alias="AGENT_MAX_RETRIES")

    # BM25 & Chunking Constraints
    bm25_top_links: int = Field(default=25, alias="BM25_TOP_LINKS")
    bm25_top_chunks: int = Field(default=5, alias="BM25_TOP_CHUNKS")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    # Safety & Fetch Settings
    contact_email: str = Field(default="agent@example.com", alias="CONTACT_EMAIL")
    user_agent: str = Field(
        default="WebRetrievalBot/1.0 (+https://github.com/amd-ai-league; agent@example.com)",
        alias="USER_AGENT",
    )
    max_page_size_bytes: int = Field(default=5_242_880, alias="MAX_PAGE_SIZE_BYTES")  # 5MB
    rate_limit_delay_seconds: float = Field(default=0.5, alias="RATE_LIMIT_DELAY_SECONDS")
    allow_external_domains: bool = Field(default=False, alias="ALLOW_EXTERNAL_DOMAINS")
    enable_playwright_fallback: bool = Field(default=False, alias="ENABLE_PLAYWRIGHT_FALLBACK")

    # HTTP API
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")

    def safe_repr(self) -> dict:
        """Return dict representation with sensitive secrets masked."""
        data = self.model_dump()
        if self.llm_api_key and self.llm_api_key != "EMPTY":
            data["llm_api_key"] = f"{self.llm_api_key[:3]}...***"
        return data


# Global singleton instance
settings = Settings()
