import os
from pathlib import Path
from functools import lru_cache
from typing import ClassVar

from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


dotenv_path = Path(__file__).resolve().parents[2] / '.env'
load_dotenv(dotenv_path)

APP_HOME = os.environ.get('APP_HOME')

class AppConfig(BaseModel):
    """Application configurations."""

    # Defines the root directory of the application.
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # Defines the settings directory located in the root directory.
    SETTINGS_DIR: Path = BASE_DIR.joinpath('settings')
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Global configurations."""

    APP_CONFIG: AppConfig = AppConfig()
    MAX_TICKER_SYMBOLS_WS: int = int(20)
    UPDATE_INTERVAL_EQUITY_DETAIL_WS: int = int(10)

    AES_ENCRYPTION_KEY: str
    FMP_API_KEY: str = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_PLACEHOLDER")
    BASE_FMP_URL: str = "https://financialmodelingprep.com/api"
    FMP_URL_STABLE: str = "https://financialmodelingprep.com/stable"
    ALCHEMY_API_KEY: str
    ETHERSCAN_API_KEY: str
    TWITTER_API_IO_KEY: str
    
    # Config Redis database
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", '')
    REDIS_DB: int = int(os.getenv("REDIS_DB", 3))
    
    # TTL (Time-To-Live) for cache, calculate seconds
    CACHE_TTL_LISTS: int = int(os.getenv("CACHE_TTL_LISTS", 60*5))
    CACHE_TTL_DETAILS: int = int(os.getenv("CACHE_TTL_DETAILS", 60 * 5))  
    CACHE_TTL_CHART: int = int(os.getenv("CACHE_TTL_CHART", 60 * 5))  
    CACHE_DEFAULT_TTL: int = int(os.getenv("CACHE_DEFAULT_TTL", 60 * 5))
    CACHE_TTL_TICKER: int = int(os.getenv("CACHE_TTL_TICKER", 60 * 1)) 
    CACHE_PROFILE: int = int(os.getenv("CACHE_TTL_DETAILS", 60 * 129600)) # 3 tháng (1 quý)
    CACHE_TTL_NEWS: int = int(os.getenv("CACHE_TTL_NEWS", 60 * 5)) 
    CACHE_TTL_LONG : int = int(os.getenv("CACHE_TTL_LONG", 60 * 60 * 24))
    CACHE_TTL_FINANCIALS_ANNUAL: int = int(os.getenv("CACHE_TTL_FINANCIALS_ANNUAL", 60 * 5)) # 30 ngày
    CACHE_TTL_FINANCIALS_QUARTERLY: int = int(os.getenv("CACHE_TTL_FINANCIALS_QUARTERLY", 60 * 5)) # 7 ngày

    CACHE_TTL_DISCOVERY_HISTORICAL: int = int(os.getenv("CACHE_TTL_DISCOVERY_HISTORICAL", 60 * 60 * 24))  # 24 giờ
    CACHE_TTL_DISCOVERY_QUOTE: int = int(os.getenv("CACHE_TTL_DISCOVERY_QUOTE", 30))  # 30 giây cho real-time quote
    
    # Define global variables with the Field class
    ENV_STATE: str = Field('dev', env='ENV_STATE')
    LOG_LEVEL: str = Field('DEBUG', env='LOG_LEVEL')

    HOST: str = Field('0.0.0.0', env='HOST')
    PORT: int = Field('8051', env='PORT')

    DEFAULT_PROVIDER: ClassVar[str] = 'fmp'
    
    # Number of workers when running Uvicorn.
    UVICORN_WORKERS: int = Field(1, env='UVICORN_WORKERS')
    UVICORN_RELOAD: str = Field("false", env='UVICORN_RELOAD')

    API_CONFIG_FILENAME: str = Field('api_config.yaml', env='API_CONFIG_FILENAME')
    LOG_CONFIG_FILENAME: str = Field('logging_config.yaml', env='LOG_CONFIG_FILENAME')
    AUTH_CONFIG_FILENAME: str = Field('auth_config.yaml', env='AUTH_CONFIG_FILENAME')
    DATABASE_CONFIG_FILENAME: str = Field('database_config.yaml', env='DATABASE_CONFIG_FILENAME')

    MODEL_CONFIG_FILENAME: str = Field('model_config.yaml', env='MODEL_CONFIG_FILENAME')

    # Define config Ollama for hosting model from local
    OLLAMA_ENDPOINT: str = Field(..., env='OLLAMA_ENDPOINT')

    # Define access token Huggingface
    HUGGINGFACE_ACCESS_TOKEN: str | None = Field(None, env='HUGGINGFACE_ACCESS_TOKEN')

    LLM_MAX_RETRIES: int = Field(5, env='LLM_MAX_RETRIES')

    # Define config for Qdrant
    QDRANT_ENDPOINT: str | None = Field(..., env='QDRANT_ENDPOINT') 
    QDRANT_COLLECTION_NAME: str = Field(..., env='QDRANT_COLLECTION_NAME')

    # MySQL Frontend config
    MYSQL_HOST: str = Field('localhost', env='MYSQL_HOST')
    MYSQL_PORT: int = Field(3306, env='MYSQL_PORT')
    MYSQL_DATABASE: str = Field('frontend_db', env='MYSQL_DATABASE')
    MYSQL_USER: str = Field('frontend_user', env='MYSQL_USER')
    MYSQL_PASSWORD: str = Field('frontend_password', env='MYSQL_PASSWORD')

    # Provider
    OLLAMA_ENDPOINT: str = Field(..., env='OLLAMA_ENDPOINT')
    OPENAI_API_KEY: str = Field("", env='OPENAI_API_KEY')
    GEMINI_API_KEY: str = Field("", env='GEMINI_API_KEY')

    MODEL_DEFAULT: str = Field("", env='MODEL_DEFAULT')
    PROVIDER_DEFAULT: str = Field("", env='PROVIDER_DEFAULT')

# Avoid having to re-read the .env file and create the Settings object every time you access it
@lru_cache()
def get_settings():
    return Settings()


# Settings will be the object that contains all the configuration of the application.
settings = get_settings()