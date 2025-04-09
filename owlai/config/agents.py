#  /\_/\
# ((@v@))
# ():::()
#  VV-VV
# OK This is the config file, this should be a regular JSON file but until I see tha value of swtiching I will keep it like this

from owlai.services.system import device, env, is_prod, is_dev, is_test

print("Loading config module")

from owlai.config.prompts import PROMPT_CONFIG
from owlai.config.tools import TOOLS_CONFIG, FRENCH_LAW_QUESTIONS

enable_multi_process = device == "cuda"

OWL_AGENTS_BASE_CONFIG = {
    "fr-law-qna": {
        "name": "fr-law-qna",
        "version": "1.0",
        "description": "Agent responsible for answering questions about french law",
        "system_prompt": PROMPT_CONFIG["marianne-v1"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "max_tokens": 2048,
            "temperature": 0.1,
            "context_size": 10000,
            "tools_names": [
                "rag-fr-general-law-v1",
                "rag-fr-tax-law-v1",
                # "rag-fr-admin-law-v1",
            ],
            "tools": {
                "rag-fr-general-law-v1": TOOLS_CONFIG["rag-fr-general-law-v1"],
                "rag-fr-tax-law-v1": TOOLS_CONFIG["rag-fr-tax-law-v1"],
            },
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"]
        + FRENCH_LAW_QUESTIONS["tax"],
    },
    "fr-law-qna-complete": {
        "name": "fr-law-qna-complete",
        "version": "1.0",
        "description": "Agent responsible for answering questions about french law",
        "system_prompt": PROMPT_CONFIG["marianne-v1"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "max_tokens": 2048,
            "temperature": 0.1,
            "context_size": 10000,
            "tools_names": ["fr-law-complete"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"]
        + FRENCH_LAW_QUESTIONS["tax"]
        + FRENCH_LAW_QUESTIONS["admin"],
    },
    "rag-naruto": {
        "name": "rag-naruto",
        "version": "1.0",
        "description": "Agent that knows everything about the anime series Naruto",
        "system_prompt": PROMPT_CONFIG["rag-en-naruto-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "max_tokens": 2000,
            "temperature": 0.3,
            "context_size": 10000,
            "tools_names": ["rag-naruto-v1"],
        },
        "default_queries": [
            "Who is Tsunade?",
            "Tell me about Orochimaru's powers.",
            "Who is the Hokage of Konoha?",
            "Tell me about sasuke's personality",
            "Who is the first sensei of naruto?",
            "what happens to the Uchiha clan?",
            "What is a sharingan?",
            "What is the akatsuki?",
            "List all the Hokage.",
        ],
    },
}


OWL_AGENTS_OPTIONAL_RAG_TOOLS = {
    "rag-naruto": {
        "name": "rag-naruto",
        "version": "1.0",
        "description": "Agent that knows everything about the anime series Naruto",
        "system_prompt": PROMPT_CONFIG["rag-en-naruto-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["rag-naruto-v1"],
        },
        "default_queries": [
            "Who is Tsunade?",
            "Tell me about Orochimaru's powers.",
            "Who is the Hokage of Konoha?",
            "Tell me about sasuke's personality",
            "Who is the first sensei of naruto?",
            "what happens to the Uchiha clan?",
            "What is a sharingan?",
            "What is the akatsuki?",
            "Who is the first Hokage?",
        ],
    },
    "rag-droit-fiscal": {
        "name": "rag-droit-fiscal",
        "version": "1.0",
        "description": "Agent specialized in french tax law. It governs the creation, collection, and control of taxes and other compulsory levies imposed by public authorities.",
        "system_prompt": PROMPT_CONFIG["rag-fr-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["rag-fr-tax-law-v1"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["tax"],
    },
    "rag-droit-admin": {
        "name": "rag-droit-admin",
        "version": "1.0",
        "description": "Agent specialized in french administrative law. It governs the organization, functioning, and accountability of public administration. It deals with the legal relationships between public authorities (e.g. the State, local governments, public institutions) and private individuals or other entities. Its core purpose is to ensure that public power is exercised lawfully and in the public interest",
        "system_prompt": PROMPT_CONFIG["rag-fr-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["rag-fr-admin-law-v1"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["admin"],
    },
}

OWL_AGENTS_PROD = {
    "fr-law-qna-complete": {
        "name": "fr-law-qna-complete",
        "version": "1.0",
        "description": "Agent responsible for answering questions about french law",
        "system_prompt": PROMPT_CONFIG["marianne-v1"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "max_tokens": 2048,
            "temperature": 0.1,
            "context_size": 10000,
            "tools_names": ["fr-law-complete"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"]
        + FRENCH_LAW_QUESTIONS["tax"]
        + FRENCH_LAW_QUESTIONS["admin"],
    },
}


OWL_AGENTS_CONFIG_ENV = {
    "development": OWL_AGENTS_BASE_CONFIG,
    "production": OWL_AGENTS_PROD,
}

# this is the hooks imported by consumers
OWL_AGENTS_CONFIG = OWL_AGENTS_CONFIG_ENV[env]

TEST_QUERIES = {
    "test_queries": [
        "remove all txt files in the temp folder.",
        "create a temp folder in the current directory if it does not exist.",
        "you must always save files in the temp folder",  # Added to the toolsystem prompt for anthropic
        "open an explorer in the temp folder",
        "get some information about the network and put it into a .txt file",
        "give me some information about the hardware and put it into a .txt file in the temp folder",
        "open the last .txt file",
        "open the bbc homepage",
        "display an owl in ascii art",
        "display an owl in ascii art and put it into a .txt file",
        # "switch off the screen for 1 second and then back on", # TODO: retest blocking the execution
        "set the brightness of the screen to 50/100",
        "list the values of the PATH environement variable in a txt file one per line",
        "open the last txt file",
        "Report all of the USB devices installed into a file",
        "print the file you saved with USB devices in the terminal",
        "set the brightness of the screen back to 100",
        "kill the notepad process",
        "display information about my network connection",
        "minimizes all windows",
        "run the keyboard combination Ctlr + Win + -> ",
    ]
}
