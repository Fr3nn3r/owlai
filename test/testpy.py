class BaseAgent:
    def __init__(self, **kwargs):
        self.config = kwargs  # Store config as a dictionary
        #print("BaseAgent initialized with:", self.config)
        print("BaseAgent initialized with:", self.config["model_provider"])

class SystemAgent(BaseAgent):
    def __init__(self, config):
        # Pass the unpacked config dictionary as keyword arguments
        super().__init__(**config)

# Example CONFIG dictionary
CONFIG = {
    "system": {
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "max_output_tokens": 200,
        "temperature": 0.5,
        "max_context_tokens": 4096,
        "tools_names": ["activate", "run_task", "play_song"],
        "system_prompt": "You are local system agent."
    }
}

# Create instance of SystemAgent
agent = SystemAgent(CONFIG["system"]["model_provider"], model_provider="openai")
