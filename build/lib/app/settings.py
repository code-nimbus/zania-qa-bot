from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    CHROMA_PERSIST_DIR: str = ".chroma"
    TOP_K: int = 4

    MAX_QUESTIONS: int = 50
    MAX_FILE_SIZE_BYTES: int = 15_000_000  # 15 MB

    OPENAI_TIMEOUT_SECONDS: int = 30
    OPENAI_MAX_OUTPUT_TOKENS: int = 300


settings = Settings()
