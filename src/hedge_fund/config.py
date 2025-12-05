import os
from dotenv import load_dotenv

load_dotenv()

# Financial data API config
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")

# Model configs
DEFAULT_MODEL_NAME = "gpt-4.1-nano"
DEFAULT_MODEL_PROVIDER = "OpenAI"

# Cache settings
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour