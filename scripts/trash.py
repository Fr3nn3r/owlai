# Fix fitz import to resolve type checking issues
try:
    from fitz import Document as PyMuPDFDocument, Page as PyMuPDFPage

    # Type aliases for type checking
    Document = PyMuPDFDocument  # type: ignore[assignment]
    Page = PyMuPDFPage  # type: ignore[assignment]
except ImportError:
    # For type hinting only
    class Document:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> "Page":
            raise NotImplementedError

        @property
        def metadata(self) -> dict:
            return {}

    class Page:
        def get_text(self, text_type: str) -> str:
            return ""


# deprecated
______RAG_AGENTS_BASE_CONFIG = [
    {
        "name": "rag-naruto-v1",
        "description": "Agent that knows everything about the anime series Naruto",
        "system_prompt": _PROMPT_CONFIG["rag-en-naruto-v1"],
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "Any question about the anime series Naruto expressed in english",
                }
            },
            "required": ["query"],
        },
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": [],
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
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "input_data_folder": "data/dataset-0001",  # Larger dataset
                "parser": {
                    "implementation": "DefaultParser",
                    "output_data_folder": "data/dataset-0001",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "add_start_index": True,
                    "strip_whitespace": True,
                    "separators": ["\n\n", "\n", " ", ""],
                    "extract_images": False,
                    "extraction_mode": "plain",
                },
            },
        },
    },
    {
        "name": "rag-fr-general-law-v1",
        "description": "Agent specialized in french civil, penal, and commercial law",
        "system_prompt": _PROMPT_CONFIG["rag-fr-v2"],
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "Any question about french civil, penal, and commercial law expressed in french",
                }
            },
            "required": ["query"],
        },
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": [],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"],
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "input_data_folder": "data/legal-rag/general",  # Larger dataset
                "parser": {
                    "implementation": "FrenchLawParser",
                    "output_data_folder": "data/legal-rag/general",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "add_start_index": True,
                    "strip_whitespace": True,
                    "separators": ["\n\n", "\n", " ", ""],
                    "extract_images": False,
                    "extraction_mode": "plain",
                },
            },
        },
    },
    {
        "name": "rag-fr-tax-law-v1",
        "description": "Agent specialized in french tax law. It governs the creation, collection, and control of taxes and other compulsory levies imposed by public authorities.",
        "system_prompt": _PROMPT_CONFIG["rag-fr-v2"],
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "Any question about french tax law expressed in french",
                }
            },
            "required": ["query"],
        },
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": [],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["tax"],
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "input_data_folder": "data/legal-rag/fiscal",  # Larger dataset
                "parser": {
                    "implementation": "FrenchLawParser",
                    "output_data_folder": "data/legal-rag/fiscal",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "add_start_index": True,
                    "strip_whitespace": True,
                    "separators": ["\n\n", "\n", " ", ""],
                    "extract_images": False,
                    "extraction_mode": "plain",
                },
            },
        },
    },
    {
        "name": "rag-fr-admin-law-v1",
        "description": "Agent specialized in french administrative law. It governs the organization, functioning, and accountability of public administration. It deals with the legal relationships between public authorities (e.g. the State, local governments, public institutions) and private individuals or other entities. Its core purpose is to ensure that public power is exercised lawfully and in the public interest",
        "system_prompt": _PROMPT_CONFIG["rag-fr-v2"],
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "Any question about french administrative law expressed in french",
                }
            },
            "required": ["query"],
        },
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": [],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["admin"],
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "input_data_folder": "data/legal-rag/admin",  # Larger dataset
                "parser": {
                    "implementation": "FrenchLawParser",
                    "output_data_folder": "data/legal-rag/admin",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "add_start_index": True,
                    "strip_whitespace": True,
                    "separators": ["\n\n", "\n", " ", ""],
                    "extract_images": False,
                    "extraction_mode": "plain",
                },
            },
        },
    },
]


_DEPRECATED_AGENTS_CONFIG = {
    "system": {
        "name": "system",
        "version": "1.0",
        "description": "Agent controlling the local system",
        "system_prompt": _PROMPT_CONFIG["system-v1"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-3.5-turbo",
            "max_tokens": 200,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["owl_system_interpreter", "play_song"],
        },
        "default_queries": [
            "list the current directory.",
            "welcome mode",
            "play Shoot to Thrill by AC/DC",
            "display an owl in ascii art",
            "open an explorer in the temp folder",
            "get some information about the network and put it into a .txt file",
            "give me some information about the hardware and put it into a .txt file in the temp folder",
            "open the bbc homepage",
            "open the last txt file in the temp folder",
            "kill the notepad process",
        ],
    },
    "identification": {
        "name": "identification",
        "version": "1.0",
        "description": "Agent responsible for identifying the user",
        "system_prompt": _PROMPT_CONFIG["identification-v1"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 200,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["security_tool"],
        },
        "default_queries": [
            "hey hi",
            "who are you?",
            "what is your goal?",
            "how can I identify myself?",
            "what is my password?",
            "what is your name?",
            "how many attempts are allowed?",
            "how many attempts do I have left?",
            "my password is red unicorn",
            "my password is pink dragon",
            "welcome mode",
            "system mode",
        ],
        "test_queries": [],
    },
    "welcome": {
        "name": "welcome",
        "version": "1.0",
        "description": "Agent responsible for welcoming the user",
        "system_prompt": _PROMPT_CONFIG["welcome-v1"],
        "llm_config": {
            "model_provider": "mistralai",
            "model_name": "mistral-large-latest",
            "max_tokens": 2048,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["tavily_search_results_json"],
        },
        "default_queries": [
            "system mode",
            "qna mode",
            "respond to me in french from now on",
            "who are you?",
            "what is your goal",
            "who am I?",
            "what is my name?",
            "what is my password?",
            "what is my favorite color?",
            "what is my favorite animal?",
            "what is my favorite food?",
            "what is my favorite drink?",
        ],
    },
}
