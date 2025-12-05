import os
import json
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from enum import Enum
from pydantic import BaseModel
from typing import Tuple, List, Optional
from pathlib import Path


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""

    ALIBABA = "Alibaba"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    GOOGLE = "Google"
    GROQ = "Groq"
    META = "Meta"
    MISTRAL = "Mistral"
    OPENAI = "openai"  # Keep lowercase to match JSON
    OLLAMA = "Ollama"
    OPENROUTER = "OpenRouter"
    GIGACHAT = "GigaChat"
    AZURE_OPENAI = "Azure OpenAI"
    XAI = "xAI"


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""

    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)

    def is_custom(self) -> bool:
        """Check if the model is a custom model"""
        return self.model_name == "-"

    def has_json_mode(self) -> bool:
        """Check if the model supports JSON mode"""
        if self.is_deepseek() or self.is_gemini():
            return False
        # Only certain Ollama models support JSON mode
        if self.is_ollama():
            return "llama3" in self.model_name or "neural-chat" in self.model_name
        # OpenRouter models generally support JSON mode
        if self.provider == ModelProvider.OPENROUTER:
            return True
        return True

    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_gemini(self) -> bool:
        """Check if the model is a Gemini model"""
        return self.model_name.startswith("gemini")

    def is_ollama(self) -> bool:
        """Check if the model is an Ollama model"""
        return self.provider == ModelProvider.OLLAMA


def normalize_provider_name(provider_str: str) -> str:
    """
    Normalize provider name to match enum values.
    Handles case-insensitive matching and common variations.
    """
    # Create a mapping of lowercase names to actual enum values
    provider_map = {
        "alibaba": ModelProvider.ALIBABA,
        "anthropic": ModelProvider.ANTHROPIC,
        "deepseek": ModelProvider.DEEPSEEK,
        "google": ModelProvider.GOOGLE,
        "groq": ModelProvider.GROQ,
        "meta": ModelProvider.META,
        "mistral": ModelProvider.MISTRAL,
        "openai": ModelProvider.OPENAI,
        "ollama": ModelProvider.OLLAMA,
        "openrouter": ModelProvider.OPENROUTER,
        "gigachat": ModelProvider.GIGACHAT,
        "azure_openai": ModelProvider.AZURE_OPENAI,
        "azure openai": ModelProvider.AZURE_OPENAI,
        "xai": ModelProvider.XAI,
    }
    
    # Convert to lowercase and remove extra spaces
    normalized = provider_str.lower().strip()
    
    # Look up the correct enum value
    if normalized in provider_map:
        return provider_map[normalized]
    
    # If not found, try to match by partial name
    for key, value in provider_map.items():
        if key in normalized or normalized in key:
            return value
    
    # If still not found, raise an error with helpful message
    raise ValueError(
        f"Unknown provider: '{provider_str}'. "
        f"Valid providers are: {', '.join(provider_map.keys())}"
    )


def load_models_from_json(json_path: str) -> List[LLMModel]:
    """Load models from a JSON file with robust provider handling"""
    with open(json_path, 'r') as f:
        models_data = json.load(f)
    
    models = []
    for model_data in models_data:
        try:
            # Normalize the provider string to match enum
            provider_enum = normalize_provider_name(model_data["provider"])
            models.append(
                LLMModel(
                    display_name=model_data["display_name"],
                    model_name=model_data["model_name"],
                    provider=provider_enum
                )
            )
        except ValueError as e:
            print(f"Warning: Skipping model {model_data.get('model_name', 'unknown')}: {e}")
            continue
    
    return models


# Get the path to the JSON files
current_dir = Path(__file__).parent
models_json_path = current_dir / "api_models.json"
ollama_models_json_path = current_dir / "ollama_models.json"

# Load available models from JSON
AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# Load Ollama models from JSON if file exists
OLLAMA_MODELS = []
if ollama_models_json_path.exists():
    OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# Create Ollama LLM_ORDER separately
OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


def get_model_info(model_name: str, model_provider: str) -> Optional[LLMModel]:
    """Get model information by model_name and provider"""
    all_models = AVAILABLE_MODELS + OLLAMA_MODELS
    
    # Normalize the provider string for comparison
    try:
        normalized_provider = normalize_provider_name(model_provider)
    except ValueError:
        return None
    
    return next(
        (model for model in all_models 
         if model.model_name == model_name and model.provider == normalized_provider),
        None
    )


def get_models_list():
    """Get the list of models for API responses."""
    return [
        {
            "display_name": model.display_name,
            "model_name": model.model_name,
            "provider": model.provider.value
        }
        for model in AVAILABLE_MODELS
    ]


def _get_ollama_base_url() -> str:
    """
    Get Ollama base URL.
    """
    # Check for custom base URL
    base_url = os.getenv("OLLAMA_BASE_URL")
    if base_url:
        return base_url
    
    # Check for OLLAMA_HOST
    host = os.getenv("OLLAMA_HOST")
    if host:
        # Remove protocol if present
        host = host.replace('http://', '').replace('https://', '')
        # Remove port if present (we'll add it back)
        host = host.split(':')[0]
        return f"http://{host}:11445"
    
    # Check if we're in Docker
    is_docker = os.path.exists('/.dockerenv')
    
    if is_docker:
        # In Docker, use host.docker.internal to reach host machine
        return "http://host.docker.internal:11434"
    else:
        # Not in Docker, use localhost
        return "http://localhost:11434"

def get_model(model_name: str, model_provider: ModelProvider, api_keys: dict = None) -> Optional[ChatOpenAI]:
    """
    Get a model instance based on name and provider.
    
    Supports:
    - OpenAI (via ChatOpenAI)
    - Ollama (via ChatOllama)
    
    Args:
        model_name: Name of the model (e.g., "gpt-4o", "llama3.1:70b")
        model_provider: ModelProvider enum value
        api_keys: Optional dict of API keys
        
    Returns:
        LangChain chat model instance
        
    Raises:
        ValueError: If provider is not implemented or API key is missing
    """
    if model_provider == ModelProvider.OPENAI:
        # Get and validate API key
        api_key = (api_keys or {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
            raise ValueError("OpenAI API key not found. Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
        
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url if base_url else None
        )
    
    elif model_provider == ModelProvider.OLLAMA:
        # Get Ollama base URL (handles Docker automatically)
        base_url = _get_ollama_base_url()
        
        print(f"Initializing Ollama with model '{model_name}' at {base_url}")
        
        try:
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=0.1,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Ollama model '{model_name}'. "
                f"Make sure Ollama is running at {base_url}. "
                f"Error: {str(e)}"
            )
    
    # Add other providers here as needed
    else:
        raise ValueError(
            f"Provider {model_provider} is not currently implemented. "
            f"Supported providers: OpenAI, Ollama"
        )


# import os
# import json
# # from langchain_anthropic import ChatAnthropic
# # from langchain_deepseek import ChatDeepSeek
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_groq import ChatGroq
# # from langchain_xai import ChatXAI
# # from langchain_openai import ChatOpenAI , AzureChatOpenAI
# from langchain_openai import ChatOpenAI
# # from langchain_gigachat import GigaChat
# # from langchain_ollama import ChatOllama
# from enum import Enum
# from pydantic import BaseModel
# from typing import Tuple, List
# from pathlib import Path


# class ModelProvider(str, Enum):
#     """Enum for supported LLM providers"""

#     ALIBABA = "Alibaba"
#     ANTHROPIC = "Anthropic"
#     DEEPSEEK = "DeepSeek"
#     GOOGLE = "Google"
#     GROQ = "Groq"
#     META = "Meta"
#     MISTRAL = "Mistral"
#     OPENAI = "openai"
#     OLLAMA = "Ollama"
#     OPENROUTER = "OpenRouter"
#     GIGACHAT = "GigaChat"
#     AZURE_OPENAI = "Azure OpenAI"
#     XAI = "xAI"


# class LLMModel(BaseModel):
#     """Represents an LLM model configuration"""

#     display_name: str
#     model_name: str
#     provider: ModelProvider

#     def to_choice_tuple(self) -> Tuple[str, str, str]:
#         """Convert to format needed for questionary choices"""
#         return (self.display_name, self.model_name, self.provider.value)

#     def is_custom(self) -> bool:
#         """Check if the model is a Gemini model"""
#         return self.model_name == "-"

#     def has_json_mode(self) -> bool:
#         """Check if the model supports JSON mode"""
#         if self.is_deepseek() or self.is_gemini():
#             return False
#         # Only certain Ollama models support JSON mode
#         if self.is_ollama():
#             return "llama3" in self.model_name or "neural-chat" in self.model_name
#         # OpenRouter models generally support JSON mode
#         if self.provider == ModelProvider.OPENROUTER:
#             return True
#         return True

#     def is_deepseek(self) -> bool:
#         """Check if the model is a DeepSeek model"""
#         return self.model_name.startswith("deepseek")

#     def is_gemini(self) -> bool:
#         """Check if the model is a Gemini model"""
#         return self.model_name.startswith("gemini")

#     def is_ollama(self) -> bool:
#         """Check if the model is an Ollama model"""
#         return self.provider == ModelProvider.OLLAMA


# # Load models from JSON file
# def load_models_from_json(json_path: str) -> List[LLMModel]:
#     """Load models from a JSON file"""
#     with open(json_path, 'r') as f:
#         models_data = json.load(f)
    
#     models = []
#     for model_data in models_data:
#         # Convert string provider to ModelProvider enum
#         provider_enum = ModelProvider(model_data["provider"])
#         models.append(
#             LLMModel(
#                 display_name=model_data["display_name"],
#                 model_name=model_data["model_name"],
#                 provider=provider_enum
#             )
#         )
#     return models


# # Get the path to the JSON files
# current_dir = Path(__file__).parent
# models_json_path = current_dir / "api_models.json"
# ollama_models_json_path = current_dir / "ollama_models.json"

# # Load available models from JSON
# AVAILABLE_MODELS = load_models_from_json(str(models_json_path))

# # Load Ollama models from JSON
# OLLAMA_MODELS = load_models_from_json(str(ollama_models_json_path))

# # Create LLM_ORDER in the format expected by the UI
# LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

# # Create Ollama LLM_ORDER separately
# OLLAMA_LLM_ORDER = [model.to_choice_tuple() for model in OLLAMA_MODELS]


# def get_model_info(model_name: str, model_provider: str) -> LLMModel | None:
#     """Get model information by model_name"""
#     all_models = AVAILABLE_MODELS + OLLAMA_MODELS
#     return next((model for model in all_models if model.model_name == model_name and model.provider == model_provider), None)


# def get_models_list():
#     """Get the list of models for API responses."""
#     return [
#         {
#             "display_name": model.display_name,
#             "model_name": model.model_name,
#             "provider": model.provider.value
#         }
#         for model in AVAILABLE_MODELS
#     ]


# def get_model(model_name: str, model_provider: ModelProvider, api_keys: dict = None) -> ChatOpenAI | None:
#     # if model_provider == ModelProvider.GROQ:
#     #     api_key = (api_keys or {}).get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
#     #     if not api_key:
#     #         # Print error to console
#     #         print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file or provided via API keys.")
#     #         raise ValueError("Groq API key not found.  Please make sure GROQ_API_KEY is set in your .env file or provided via API keys.")
#     #     return ChatGroq(model=model_name, api_key=api_key)
#     if model_provider == ModelProvider.OPENAI:
#         # Get and validate API key
#         api_key = (api_keys or {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
#         base_url = os.getenv("OPENAI_API_BASE")
#         if not api_key:
#             # Print error to console
#             print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
#             raise ValueError("OpenAI API key not found.  Please make sure OPENAI_API_KEY is set in your .env file or provided via API keys.")
#         return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
#     # elif model_provider == ModelProvider.ANTHROPIC:
#     #     api_key = (api_keys or {}).get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
#     #     if not api_key:
#     #         print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file or provided via API keys.")
#     #         raise ValueError("Anthropic API key not found.  Please make sure ANTHROPIC_API_KEY is set in your .env file or provided via API keys.")
#     #     return ChatAnthropic(model=model_name, api_key=api_key)
#     # elif model_provider == ModelProvider.DEEPSEEK:
#     #     api_key = (api_keys or {}).get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
#     #     if not api_key:
#     #         print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file or provided via API keys.")
#     #         raise ValueError("DeepSeek API key not found.  Please make sure DEEPSEEK_API_KEY is set in your .env file or provided via API keys.")
#     #     return ChatDeepSeek(model=model_name, api_key=api_key)
#     # elif model_provider == ModelProvider.GOOGLE:
#     #     api_key = (api_keys or {}).get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
#     #     if not api_key:
#     #         print(f"API Key Error: Please make sure GOOGLE_API_KEY is set in your .env file or provided via API keys.")
#     #         raise ValueError("Google API key not found.  Please make sure GOOGLE_API_KEY is set in your .env file or provided via API keys.")
#     #     return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
#     # elif model_provider == ModelProvider.OLLAMA:
#     #     # For Ollama, we use a base URL instead of an API key
#     #     # Check if OLLAMA_HOST is set (for Docker on macOS)
#     #     ollama_host = os.getenv("OLLAMA_HOST", "localhost")
#     #     base_url = os.getenv("OLLAMA_BASE_URL", f"http://{ollama_host}:11434")
#     #     return ChatOllama(
#     #         model=model_name,
#     #         base_url=base_url,
#     #     )
#     # elif model_provider == ModelProvider.OPENROUTER:
#     #     api_key = (api_keys or {}).get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
#     #     if not api_key:
#     #         print(f"API Key Error: Please make sure OPENROUTER_API_KEY is set in your .env file or provided via API keys.")
#     #         raise ValueError("OpenRouter API key not found. Please make sure OPENROUTER_API_KEY is set in your .env file or provided via API keys.")
        
#         # Get optional site URL and name for headers
#     #     site_url = os.getenv("YOUR_SITE_URL", "https://github.com/virattt/ai-hedge-fund")
#     #     site_name = os.getenv("YOUR_SITE_NAME", "AI Hedge Fund")
        
#     #     return ChatOpenAI(
#     #         model=model_name,
#     #         openai_api_key=api_key,
#     #         openai_api_base="https://openrouter.ai/api/v1",
#     #         model_kwargs={
#     #             "extra_headers": {
#     #                 "HTTP-Referer": site_url,
#     #                 "X-Title": site_name,
#     #             }
#     #         }
#     #     )
#     # elif model_provider == ModelProvider.XAI:
#     #     api_key = (api_keys or {}).get("XAI_API_KEY") or os.getenv("XAI_API_KEY")
#     #     if not api_key:
#     #         print(f"API Key Error: Please make sure XAI_API_KEY is set in your .env file or provided via API keys.")
#     #         raise ValueError("xAI API key not found. Please make sure XAI_API_KEY is set in your .env file or provided via API keys.")
#     #     return ChatXAI(model=model_name, api_key=api_key)
#     # elif model_provider == ModelProvider.GIGACHAT:
#     #     if os.getenv("GIGACHAT_USER") or os.getenv("GIGACHAT_PASSWORD"):
#     #         return GigaChat(model=model_name)
#     #     else: 
#     #         api_key = (api_keys or {}).get("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_API_KEY") or os.getenv("GIGACHAT_CREDENTIALS")
#     #         if not api_key:
#     #             print("API Key Error: Please make sure api_keys is set in your .env file or provided via API keys.")
#     #             raise ValueError("GigaChat API key not found. Please make sure GIGACHAT_API_KEY is set in your .env file or provided via API keys.")

#     #         return GigaChat(credentials=api_key, model=model_name)
#     # elif model_provider == ModelProvider.AZURE_OPENAI:
#     #     # Get and validate API key
#     #     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     #     if not api_key:
#     #         # Print error to console
#     #         print(f"API Key Error: Please make sure AZURE_OPENAI_API_KEY is set in your .env file.")
#     #         raise ValueError("Azure OpenAI API key not found.  Please make sure AZURE_OPENAI_API_KEY is set in your .env file.")
#     #     # Get and validate Azure Endpoint
#     #     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     #     if not azure_endpoint:
#     #         # Print error to console
#     #         print(f"Azure Endpoint Error: Please make sure AZURE_OPENAI_ENDPOINT is set in your .env file.")
#     #         raise ValueError("Azure OpenAI endpoint not found.  Please make sure AZURE_OPENAI_ENDPOINT is set in your .env file.")
#     #     # get and validate deployment name
#     #     azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
#     #     if not azure_deployment_name:
#     #         # Print error to console
#     #         print(f"Azure Deployment Name Error: Please make sure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.")
#     #         raise ValueError("Azure OpenAI deployment name not found.  Please make sure AZURE_OPENAI_DEPLOYMENT_NAME is set in your .env file.")
#     #     return AzureChatOpenAI(azure_endpoint=azure_endpoint, azure_deployment=azure_deployment_name, api_key=api_key, api_version="2024-10-21")
