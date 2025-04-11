from owlai.services.system import device

enable_multi_process = device == "cuda"

FRENCH_LAW_QUESTIONS = {
    "general": [
        "Quels sont les délais d'obtention d'un permis de séjour en France ?",
        "Quelles sont les démarches à entreprendre pour obtenir un titre de séjour pour soins ?",
        "Quelles sont les conditions à remplir pour obtenir un titre de séjour pour soins ?",
        "Quelle pension alimentaire dois-je verser à ma fille de 21 ans qui est étudiante et a un revenu de 1000 euros par mois ?",
        "Je suis en litige avec mon employeur et la sécurité sociale sur une contestation de mon accident du travail, que dois-je faire ?",
        "Puis-je obtenir un titre de séjour pour soins pour de l'urticaire ?",
        "Quelles pathologies concernent un titre de séjour pour soins ?",
        "Expliquez la gestion en France de la confusion des peines.",
        "Dans quelles conditions un propriétaire est-il responsable des dommages causés par son animal domestique ?",
        "Quels sont les critères pour invoquer la nullité d'un contrat pour vice du consentement ?",
        "Quelle est la différence entre un vol simple et un vol aggravé en droit pénal français ?",
        "Quelle est la peine maximale encourue pour abus de confiance selon le code pénal ?",
        "Combien de temps peut durer une garde à vue en droit français, et sous quelles conditions peut-elle être prolongée ?",
        "À quel moment un avocat peut-il accéder au dossier pénal d'un suspect durant une enquête ?",
        "Quelles sont les principales différences entre une SARL et une SAS en droit commercial français ?",
        "Dans quelles conditions peut-on engager une procédure de redressement judiciaire pour une entreprise en difficulté ?",
        "Quelles sont les conditions de validité d'un licenciement pour faute grave ?",
        "Quelle est la durée légale du congé maternité en France, selon le code du travail ?",
        "Citez l'article 1243 du code civil.",
    ],
    "tax": [
        "Quels revenus sont exonérés d'impôt sur le revenu selon le Code général des impôts ?",
        "Quelle est la procédure à suivre en cas de désaccord avec un redressement fiscal notifié par l'administration fiscale ?",
        "Quels taux de TVA s'appliquent à la restauration en France ?",
        "Dans quelles conditions peut-on bénéficier d'un crédit d'impôt pour travaux de rénovation énergétique ?",
        "Quelle est la différence entre l'évasion fiscale et la fraude fiscale en droit français ?",
        "Quelles sont les principales obligations fiscales d'une entreprise française qui exporte hors de l'Union européenne ?",
        "Quel est le délai de prescription en matière de contrôle fiscal des particuliers ?",
        "Comment se calcule la Contribution Sociale Généralisée (CSG) sur les revenus du patrimoine ?",
        "Quelles collectivités locales sont habilitées à prélever une taxe foncière, selon le Code général des collectivités territoriales ?",
        "Quelles taxes spécifiques s'appliquent sur les carburants selon le Code des impositions sur les biens et services ?",
    ],
    "admin": [
        "Quelle juridiction administrative est compétente en première instance pour contester un permis de construire ?",
        "Quelles sont les principales étapes d'une procédure devant le tribunal administratif ?",
        "Sous quelles conditions une collectivité territoriale peut-elle conclure un marché public sans mise en concurrence préalable ?",
        "Quelle procédure doit suivre une commune pour vendre un bien immobilier lui appartenant ?",
        "Quels documents sont nécessaires pour obtenir un permis d'aménager selon le Code de l'urbanisme ?",
        "Dans quels cas une étude d'impact environnementale est-elle obligatoire pour un projet d'infrastructure publique ?",
        "Quelles sont les conditions légales pour qu'une expropriation pour cause d'utilité publique soit valide ?",
        "Quels délais doit respecter une collectivité pour répondre à une demande d'accès à un document administratif ?",
        "Dans quel cas une décision administrative peut-elle faire l'objet d'un référé-suspension devant le juge administratif ?",
        "Quelles sanctions administratives une entreprise encourt-elle en cas de manquement grave à un marché public ?",
    ],
}


FR_LAW_PARSER_CONFIG = {
    "implementation": "FrenchLawParser",
    "output_data_folder": "data/legal-fr-complete/images",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "add_start_index": True,
    "strip_whitespace": True,
    "separators": ["\n\n", "\n", " ", ""],
    "extract_images": False,
    "extraction_mode": "plain",
}

DEFAULT_PARSER_CONFIG = {
    "output_data_folder": "data/legal-fr-complete",
    "chunk_size": 512,
    "chunk_overlap": 100,
    "add_start_index": True,
    "strip_whitespace": True,
    "separators": ["\n\n", "\n", ".", " ", ""],
    "extract_images": False,
    "extraction_mode": "plain",
}


TOOLS_CONFIG = {
    "rag-fr-admin-law-v1": {
        "name": "rag-fr-admin-law-v1",
        "description": "Returns data chunks from french administration law documents",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french administration law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "name": "rag-fr-admin-law",
                "version": "0.3.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/cache/rag-fr-admin-law",  # Larger dataset
                "parser": FR_LAW_PARSER_CONFIG,
            },
        },
    },
    "rag-fr-tax-law-v1": {
        "name": "rag-fr-tax-law-v1",
        "description": "Returns data chunks from french tax law documents",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french tax law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "name": "rag-fr-tax-law",
                "version": "0.3.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/legal-rag/fiscal",  # Larger dataset
                "parser": FR_LAW_PARSER_CONFIG,
            },
        },
    },
    "rag-fr-general-law-v1": {
        "name": "rag-fr-general-law-v1",
        "description": "Returns data chunks from french law documents: civil, work, commercial, criminal, residency, social, public health",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "name": "rag-fr-general-law",
                "version": "0.3.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/legal-rag/general",  # Larger dataset
                "parser": FR_LAW_PARSER_CONFIG,
            },
        },
    },
    "fr-law-complete": {
        "name": "fr-law-complete",
        "description": "Returns data chunks from french law documents: civil, work, commercial, criminal, residency, social, public health",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french law expressed in french",
                }
            },
            "required": ["query"],
        },
        "retriever": {
            "num_retrieved_docs": 30,
            "num_docs_final": 5,
            "embeddings_model_name": "thenlper/gte-small",
            "reranker_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "model_kwargs": {"device": device},
            "encode_kwargs": {"normalize_embeddings": True},
            "multi_process": enable_multi_process,
            "datastore": {
                "name": "rag-fr-law-complete",
                "version": "0.0.1",
                "cache_data_folder": "data/cache",
                "input_data_folder": "data/cache/rag-fr-law-complete",
                "parser": DEFAULT_PARSER_CONFIG,
            },
        },
    },
    "pinecone_french_law_lookup": {
        "name": "pinecone_french_law_lookup",
        "description": "Returns data chunks from french law documents.",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "A request for semantic search on french law expressed in french",
                }
            },
            "required": ["query"],
        },
        "num_retrieved_docs": 5,
    },
}


OPTIONAL_TOOLS = {
    "rag-naruto-v1": {
        "name": "rag-naruto-v1",
        "description": "Tool specialized in the anime naruto",
        "args_schema": {
            "title": "ToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "Any question about the anime naruto expressed in english",
                }
            },
            "required": ["query"],
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
                "input_data_folder": "data/naruto-complete",  # Larger dataset
                "cache_data_folder": "data/cache",
                "version": "0.0.1",
                "name": "naruto-complete",
                "parser": {
                    "output_data_folder": "data/naruto-complete",
                },
            },
        },
    },
    "tavily_search_results_json": {
        "max_results": 2,
    },
    "security_tool": {
        "name": "security_tool",
        "description": "A tool to check the security of the system. "
        "Useful for when you need to identify a user by password. "
        "Input should be a password.",
        "schema_params": {
            "title": "SecurityToolInput",
            "type": "object",
            "properties": {
                "query": {
                    "title": "Query",
                    "type": "string",
                    "description": "a password (sequence of words separated by spaces)",
                }
            },
            "required": ["query"],
        },
    },
}
