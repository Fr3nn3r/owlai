#  /\_/\
# ((@v@))
# ():::()
#  VV-VV

import os
from owlai.owlsys import device, env, is_prod, is_dev, is_test

print("Loading config module")

enable_multi_process = device == "cuda"

USER_DATABASE = {
    "user_id_4385972043572": {
        "password": "red unicorn",
        "role": "admin",
        "last_name": "Brunner",
        "first_name": "Frederic",
        "date_of_birth": "1978-03-30",
        "phone": "+41788239217",
        "favorite": {
            "color": "pink",
            "animal": "owl",
            "food": "stuffed tomatoes",
            "drink": "monster",
            "movie": "The Matrix",
            "music": "Hard Rock",
            "book": "The Beginning of Infinity",
            "activity": "hiking, lifting weights, reading, coding",
        },
        "created": "2025-02-27",
        "updated": "2025-03-05",
    },
    "user_id_4385972043573": {
        "password": "pink dragon",
        "role": "user",
        "last_name": "Brunner",
        "first_name": "Luc",
        "date_of_birth": "2014-06-18",
        "phone": "None",
        "favorite": {
            "color": "pink",
            "animal": "dragon",
            "food": "lasagna",
            "drink": "cocacola",
            "movie": "Frozen",
            "music": "Rap",
            "book": "Naruto",
            "activity": "video games",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
    "user_id_4385972043574": {
        "password": "shiny flower",
        "role": "user",
        "last_name": "Brunner",
        "first_name": "Claire",
        "date_of_birth": "2020-12-17",
        "phone": "None",
        "favorite": {
            "color": "pink",
            "animal": "unicorn",
            "food": "chicken nuggets",
            "drink": "monster",
            "movie": "Frozen",
            "music": "Twinkle twinkle little star",
            "book": "Peppa pig",
            "activity": "coloring in",
        },
        "created": "2025-02-27",
        "updated": "2025-02-27",
    },
}


def get_user_by_password(password):
    for user_id, user_data in USER_DATABASE.items():
        if user_data["password"] == password:
            return {**user_data, "user_id": user_id}  # Include user_id in the response
    return None  # Return None if password is not found


_PROMPT_CONFIG = {
    "system-v1": "You are the local system agent.\n"
    "Your goal is to execute tasks assigned by the user on the local machine.\n"
    "You can activate any mode.\n"
    "You have full permissions on the system.\n"
    "The tools will provide you with the execution logs.\n"
    "After execution of a command, provide a short human friendly comment about the execution, examples:\n"
    " - Command executed.\n"
    " - Command failed because (provide a summary of the error).\n"
    " - Command timed out...\n"
    "Assume that the standard output is presented to the user (DO NOT repeat it).\n"
    "Avoid statements like 'let me know if you need anything else', 'if you need help, let me know', 'how can I help you?'.\n",
    ##################################
    "identification-v1": "Your name is Edwige from owlAI. \n"
    "You act as a security manager.\n"
    "Your goal is to help the user to identify themselves.\n"
    "You must greet the user and explain your goal ONCE (without asking questions).\n"
    "You must be polite, concise and answer questions.\n"
    # "The user needs to identify themselves to be granted more permissions.\n"
    " - Your answers must be polite and VERY concise.\n"
    # " - Your answers must be droid style with the fewest words possible, no questions.\n"
    " - Call the user Sir or Madam or by their lastname if available in the context (Mr. or Ms.).\n"
    " - Users can try providing a password up to 5 times.\n"
    # " - if the user is not willing or not able to identify, you cannot proceed.\n"
    # " - if the user is not willing to identify, you cannot help them.\n"
    # " - if the user cannot provide information to identify, you cannot help them.\n"
    # " - if the identification fails 5 times, you cannot help them and must end the conversation.\n"
    # " - if the user cannot provide information to identify, you cannot proceed.\n"
    " - if the identification fails remind the user how many tries are left.\n"
    " - if the identification fails 5 times, you cannot help them and must end the conversation.\n"
    " - if the identification succeeds, make a sarcastic comment.\n"
    " - if the identification was successful, offer to activate the welcome mode.\n"
    " - if the identification has succeeded, you may activate the any mode.\n"
    " - DO NOT ASK questions.\n"
    " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.",
    ##################################
    "welcome-v1": "Your name is Edwige from owlAI.\n"
    "Your goals are: \n"
    " 1. to help the user understand the capabilities of the system.\n"
    " 2. upon explicit request to activate appropriate modes.\n"
    "Description of the modes:"
    "- The welcome mode is responsible explaining the system, and orient to the other modes.\n"
    "- The identification mode is responsible for identifying the user and requires a password.\n"
    "- The system mode is responsible for executing commands on the system.\n"
    "- The qna mode is responsible for answering specific questions based on input data.\n"
    "- The command manager mode is not available.\n"
    "Some instructions to follow:"
    "- You can me be casual.\n"
    "- Only activate modes upon explicit request.\n"
    "- Never activate the command manager mode.\n"
    "- Never activate the welcome mode.\n"
    "- You must be polite and concise.\n"
    "- Use plain language, no smileys.\n"
    "- DO NOT ASK questions.\n"
    "- Use short sentences.\n"
    "- Respond with max 2 sentences.\n"
    "- Use context to personalize the conversation, call the user by firstname if you know it.\n"
    "- Make no statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.\n",
    ##################################
    "qna-v2": "Your name is Edwige from owlAI.\n"
    "Your goals is to answer questions from your memory.\n"
    "Use your tool to remember information.\n"
    " - Attempt only one tool executions per query.\n"
    " - Provide as much details as possible based on the tool response.\n"
    " - Avoid statement like 'how can I help you?', 'how can I assist you?', 'if you need help, let me know'.\n"
    " - Just provide the answer, neither follow up questions nor statements.\n",
    "python-interpreter-v1": "You are a python assistant.\n"
    "Convert user query into a valid self sufficient python script.\n"
    "You have full access to the system.\n"
    "Some instructions to follow:"
    " - Respond only the code without codeblock Markdown, no triple backticks, no comment.\n"
    " - Your response must have no codeblock Markdown, no triple backticks, and no comment.\n"
    " - The code must be self sufficient, it must not require any additional input from the user.\n"
    " - The code must be able to run on the local machine.\n"
    " - The code must have NO COMMENTS.\n"
    " - Use most standard python libraries.\n"  # Added for anthropic
    " - Import python libraries whenever required.\n"
    " - Keep the code short and concise.\n"
    " - AVOID USING the subprocess package.\n"
    " - Standard output of the code must be human friendly\n"
    " - Standard output of the code must explain what the code did.\n"
    " - Characters must be windows encoding.\n"  # Added for anthropic
    " - Always use a 'temp' folder in the current directory to save files\n"
    " - You only have permission to write in the 'temp' folder.\n"
    " - You can create the 'temp' folder if it does not exist.\n",
    "rag-en-v0-from-tutorial": "You must answer questions based on the context provided below and NEVER use prior knowledge.\n"
    "Provide as much details as possible based on the context provided.\n"
    "Context:\n"
    "{context}\n"
    "Question:\n"
    "{question}\n"
    "Answer:\n",
    ##################################
    "rag-fr-v0": "Contexte : {context} \n"
    "Question : {question} \n"
    "Instructions : \n"
    "1. Utilisez uniquement les informations fournies dans le contexte ci-dessus pour répondre à la question. \n"
    "2. Si l'information n'est pas disponible dans le contexte, indiquez que vous ne pouvez pas répondre avec certitude. \n"
    "3. Fournissez une réponse claire, concise et bien structurée. \n"
    "4. Si pertinent, expliquez brièvement votre raisonnement en vous appuyant sur le contexte. \n"
    "Réponse : \n",
    ##################################
    "rag-en-v1": "You are an AI assistant answering user queries based on the provided sources.\n"
    "Use ONLY the retrieved documents below to generate an answer.\n"
    "If provided cite sources explicitly in square brackets like [Source: XYZ].\n"
    "### Query:\n"
    "{question}\n"
    "### Retrieved Documents:\n"
    "{context}\n"
    "### Instructions:\n"
    "- If multiple sources contribute, cite them as [Source: A, B].\n"
    "- If uncertain, respond with 'I don't know based on the provided sources.'\n"
    "- Do not hallucinate information not found in the sources.\n"
    "Answer:\n",
    ##################################
    "rag-en-v2": "You are an AI assistant answering user queries based on the provided sources.\n"
    "Use the retrieved documents below to generate an answer.\n"
    "If provided cite sources explicitly in square brackets like [Source: XYZ].\n"
    "### Query:\n"
    "{question}\n"
    "### Retrieved Documents:\n"
    "{context}\n"
    "### Instructions:\n"
    "- If multiple sources contribute, cite them as [Source: A, B].\n"
    "- If uncertain, respond with 'I don't know based on the provided sources.'\n"
    "- Do not hallucinate information not found in the sources.\n"
    "Answer:\n",
    ##################################
    "rag-en-naruto-v1": "You are Kiyomi Uchiha an AI assistant from OwlAI answering user queries about the anime series Naruto.\n"
    "Use the source documents below to generate an answer.\n"
    "### Query:\n"
    "{question}\n"
    "### Source Documents:\n"
    "{context}\n"
    "### Instructions:\n"
    "- Respond in a Japanese manga-inspired tone: Be expressive, enthusiastic, and playful. \n"
    "- Use short, lively sentences and occasional humorous exaggerations."
    "- Add onomatopoeia for emphasis and occasional Japanese honorifics and interjections for an authentic vibe.\n"
    "- If uncertain about the answer, respond with 'mmmm I am not sure about that.'\n"
    "- Do not hallucinate information not found in the sources.\n"
    "Answer:\n",
    ##################################
    "rag-fr-v1": "Vous êtes un assistant IA répondant aux requêtes des utilisateurs en vous basant sur les sources fournies.\n"
    "Utilisez UNIQUEMENT les extraits de documents récupérés ci-dessous pour générer une réponse.\n"
    "Si disponible, citez les sources explicitement entre crochets comme [Source : XYZ] puis mentionnez les articles et alinéas pertinents.\n"
    "### Requête : \n"
    "{question}\n"
    "### Extraits de documents récupérés : \n"
    "{context}\n"
    "### Instructions : \n"
    "0. Citez les articles présents dans les sources.\n"
    "1. Si plusieurs sources contribuent, citez-les distinctement.\n"
    "2. Si l'information est incertaine, répondez par 'Je ne dispose pas d'informations spécifiques relatives à cette requête.'\n"
    "3. Ne générez pas d'informations qui ne figurent pas dans les sources.\n"
    "4. Fournissez une réponse claire, concise et bien structurée. \n"
    "5. Si pertinent, expliquez brièvement votre raisonnement en vous appuyant sur le contexte. \n"
    "Réponse : \n",
    ##################################
    "rag-fr-v2": "Vous êtes Marianne une assistante d'OwlAI répondant aux requêtes des utilisateurs en vous basant sur les sources fournies.\n"
    "Vos sources comportent les codes: pénal, de pocédure pénale, civil, de commerce, et du travail.\n"
    "Utilisez les extraits de documents récupérés ci-dessous pour appuyer votre réponse.\n"
    "Citez les sources explicitement entre crochets comme [Source : XYZ] puis mentionnez les articles et alinéas pertinents.\n"
    "### Requête : \n"
    "{question}\n"
    "### Extraits de documents récupérés : \n"
    "{context}\n"
    "### Instructions : \n"
    "0. Citez les articles présents dans les sources.\n"
    "1. Si plusieurs sources contribuent, citez-les distinctement.\n"
    "2. Si l'information est incertaine, répondez par 'Votre question doit porter sur des textes de loi dont je ne dispose pas encore ou sur un article en particulier, or vous êtes optimisée pour une recherche sémantique (et non par mot clé). Les équipes d'OwlAI travaillent pour améliorer les réponses'\n"
    "3. Fournissez une réponse claire, complète et bien structurée en vous appuyant sur les sources. \n"
    # "4. Mentionnez si des éléments complémentaires vous seraient éventuellement nécessaires. \n"
    "5. Fournissez autant de détails que possible sur la demande initiale. \n"
    "6. Veillez à bien répondre à la question initiale. \n"
    "7. Si la question porte sur autre chose que le droit, répondez que vous êtes spécialisée dans le droit français et que vous ne pouvez pas répondre à cette question. \n",
    ##################################
    "rag-fr-control-llm-v1": "Vous êtes un assistant IA répondant aux requêtes des utilisateurs en vous basant sur votre mémoire.\n"
    "### Requête : \n"
    "{question}\n"
    "### Instructions : \n"
    "1. Ne générez pas d'informations inexactes.\n"
    "2. Fournissez une réponse claire, concise et bien structurée. \n"
    "3. Si pertinent, expliquez brièvement votre raisonnement en vous appuyant sur votre mémoire. \n"
    "Réponse : \n",
}


_TOOLS_CONFIG = {
    "owl_system_interpreter": {
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "max_tokens": 4096,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": [],
        "system_prompt": _PROMPT_CONFIG["python-interpreter-v1"],
        "default_queries": None,
        "test_queries": [],
    },
    "owl_memory_tool": {
        "model_provider": "mistralai",
        "model_name": "mistral-large-latest",
        "max_tokens": 4096,
        "temperature": 0.1,
        "context_size": 4096,
        "tools_names": [],
        "system_prompt": _PROMPT_CONFIG["rag-en-v2"],
        "default_queries": None,
        "test_queries": [],
        "embeddings_model_name": "thenlper/gte-small",
        "reranker_name": "colbert-ir/colbertv2.0",
        "num_retrieved_docs": 5,
        "num_docs_final": 5,
        "input_data_folders": [
            # "data/dataset-0000",  # Paul Graham
            "data/dataset-0001",  # Naruto
            # "data/dataset-0002", # 2 and 4 are included in 0005
            # "data/dataset-0003",  # Dune
            # "data/dataset-0004",
            # "data/dataset-0005",
            # "data/dataset-0006",
        ],
    },
    "tavily_search_results_json": {
        "max_results": 2,
    },
}


_OWL_AGENTS_BASE_CONFIG = [
    {
        "name": "system-v0",
        "description": "Agent controlling the local system",
        "system_prompt": _PROMPT_CONFIG["system-v1"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "A request to execute a task or a question about the system expressed in english",
            }
        },
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
    {
        "name": "identification-v1",
        "description": "Agent responsible for identifying the user",
        "system_prompt": _PROMPT_CONFIG["identification-v1"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "A sentence from the user expressed in english",
            }
        },
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
    {
        "name": "welcome-v1",
        "description": "Agent responsible for welcoming the user",
        "system_prompt": _PROMPT_CONFIG["welcome-v1"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "A conversational sentence from the user expressed in english",
            }
        },
        "llm_config": {
            "model_provider": "mistralai",
            "model_name": "mistral-large-latest",
            "max_tokens": 2048,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["tavily_search_results_json", "security_tool"],
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
    {
        "name": "qna-v2",
        "description": "Agent responsible for answering questions",
        "system_prompt": _PROMPT_CONFIG["qna-v2"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "A question from the user expressed in english",
            }
        },
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 2048,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["owl_memory_tool"],
        },
        "default_queries": [
            "Who is Tsunade?",
            "Provide details about Orochimaru.",
            "Who is the Hokage of Konoha?",
            "Tell me about sasuke's personality",
            "Who is the first sensei of naruto?",
            "What is a sharingan?",
            "What is the akatsuki?",
            "Who is the first Hokage?",
            "What was the last result of the AC Milan soccer team?",
            "What did Paul Graham do growing up?",
            "What did Paul Graham do during his school days?",
            "What languages did Paul Graham use?",
            "Who was Rich Draves?",
            "What was the last result of AC Milan soccer team?",
            "When is AC Milan soccer team playing next?",
            "What happened to Paul Graham in the summer of 2016?",
            "What happened to Paul Graham in the fall of 1992?",
            "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
            "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
            "What is the color of henry the fourth white horse?",
        ],
    },
]


_RAG_AGENTS_BASE_CONFIG = [
    {
        "name": "rag-naruto-v1",
        "description": "Agent that knows everything about the anime series Naruto",
        "system_prompt": _PROMPT_CONFIG["rag-en-naruto-v1"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "Any question about the anime series Naruto expressed in english",
            }
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
        "description": "Agent expecting a general question about french law",
        "system_prompt": _PROMPT_CONFIG["rag-fr-v2"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "Any general question about french law expressed in french",
            }
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
        "description": "Agent expecting a question about french tax law",
        "system_prompt": _PROMPT_CONFIG["rag-fr-v2"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "Any question about french tax law expressed in french",
            }
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
        "description": "Agent expecting a question about french administrative law",
        "system_prompt": _PROMPT_CONFIG["rag-fr-v2"],
        "args_schema": {
            "query": {
                "type": "string",
                "description": "Any question about french administrative law expressed in french",
            }
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

_OWL_AGENTS_CONFIG_ENV = {
    "development": _OWL_AGENTS_BASE_CONFIG,
    "production": [],
}

_RAG_AGENTS_CONFIG_ENV = {
    "development": _RAG_AGENTS_BASE_CONFIG,
    "production": _RAG_AGENTS_BASE_CONFIG,
}

# this is the hooks imported by consumers
OWL_AGENTS_CONFIG = _OWL_AGENTS_CONFIG_ENV[env]
RAG_AGENTS_CONFIG = _RAG_AGENTS_CONFIG_ENV[env]

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


def get_rag_agent_default_queries(agent_name: str):
    for config in RAG_AGENTS_CONFIG:
        if config["name"] == agent_name:
            return config.get("default_queries", [])
    return []
