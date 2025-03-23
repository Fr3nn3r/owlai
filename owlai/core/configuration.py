import os
from typing import Dict, Any, Optional
from owlai.core.config import AgentConfig, ModelConfig
from owlai.core.logging_setup import get_logger


class ConfigurationManager:
    """Manages configuration loading and environment-specific settings"""

    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.logger = get_logger("configuration")
        self._config_cache: Dict[str, Any] = {}

    def load_config(self) -> AgentConfig:
        """Load configuration for the current environment"""
        try:
            # Check cache first
            if self.environment in self._config_cache:
                return self._config_cache[self.environment]

            # Load environment-specific config
            config = self._load_environment_config()

            # Cache the config
            self._config_cache[self.environment] = config

            return config

        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def _load_environment_config(self) -> AgentConfig:
        """Load configuration specific to the current environment"""
        # Default configuration
        model_config = ModelConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=2048,
            context_size=4096,
        )

        agent_config = AgentConfig(
            name="default_agent",
            description="Default OwlAI agent",
            system_prompt="You are a helpful AI assistant.",
            llm_config=model_config,
        )

        # Override with environment-specific settings
        env_prefix = self.environment.upper()

        # Model settings
        if provider := os.getenv(f"{env_prefix}_MODEL_PROVIDER"):
            model_config.provider = provider
        if model_name := os.getenv(f"{env_prefix}_MODEL_NAME"):
            model_config.model_name = model_name
        if temp_str := os.getenv(f"{env_prefix}_MODEL_TEMPERATURE"):
            try:
                model_config.temperature = float(temp_str)
            except ValueError:
                self.logger.warning(f"Invalid temperature value: {temp_str}")
        if max_tokens_str := os.getenv(f"{env_prefix}_MODEL_MAX_TOKENS"):
            try:
                model_config.max_tokens = int(max_tokens_str)
            except ValueError:
                self.logger.warning(f"Invalid max_tokens value: {max_tokens_str}")
        if context_size_str := os.getenv(f"{env_prefix}_MODEL_CONTEXT_SIZE"):
            try:
                model_config.context_size = int(context_size_str)
            except ValueError:
                self.logger.warning(f"Invalid context_size value: {context_size_str}")

        # Agent settings
        if agent_name := os.getenv(f"{env_prefix}_AGENT_NAME"):
            agent_config.name = agent_name
        if agent_desc := os.getenv(f"{env_prefix}_AGENT_DESCRIPTION"):
            agent_config.description = agent_desc
        if system_prompt := os.getenv(f"{env_prefix}_AGENT_SYSTEM_PROMPT"):
            agent_config.system_prompt = system_prompt

        return agent_config

    def clear_cache(self) -> None:
        """Clear the configuration cache"""
        self._config_cache.clear()
