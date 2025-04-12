#  /\_/\
# ((@v@))
# ():::()
#  VV-VV
# OK This is the config file, this should be a regular JSON file but until I see tha value of swtiching I will keep it like this

from owlai.services.system import device, env, is_prod, is_dev, is_test

print("Loading config module")

from owlai.config.prompts import PROMPT_CONFIG
from owlai.config.tools import TOOLS_CONFIG

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


OWL_AGENTS_DEV = {
    "fr-law-qna-complete": {
        "name": "fr-law-qna-complete",
        "version": "1.0",
        "description": "Agent responsible for answering questions about french law",
        "system_prompt": PROMPT_CONFIG["marianne-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.1,
            "context_size": 4000,
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
            "context_size": 8000,
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
    "rag-droit-fiscal": {
        "name": "rag-droit-fiscal",
        "version": "1.0",
        "description": "Agent specialized in french tax law. It governs the creation, collection, and control of taxes and other compulsory levies imposed by public authorities.",
        "system_prompt": PROMPT_CONFIG["marianne-v2"],
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
        "description": "Agent specialized in french administrative law.",
        "system_prompt": PROMPT_CONFIG["marianne-v2"],
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
    "rag-droit-general": {
        "name": "rag-droit-general",
        "version": "1.0",
        "description": "Agent specialized in generic french law.",
        "system_prompt": PROMPT_CONFIG["marianne-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["rag-fr-general-law-v1"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"],
    },
    "rag-droit-general-pinecone": {
        "name": "rag-droit-general-pinecone",
        "version": "1.0",
        "description": "Agent specialized in generic french law.",
        "system_prompt": PROMPT_CONFIG["marianne-v2"],
        "llm_config": {
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "max_tokens": 4096,
            "temperature": 0.1,
            "context_size": 4096,
            "tools_names": ["pinecone_french_law_lookup"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"]
        + FRENCH_LAW_QUESTIONS["tax"]
        + FRENCH_LAW_QUESTIONS["admin"],
    },
}


OWL_AGENTS_OPTIONAL_RAG_TOOLS = {}

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
            "context_size": 4096,
            "tools_names": ["fr-law-complete"],
        },
        "default_queries": FRENCH_LAW_QUESTIONS["general"]
        + FRENCH_LAW_QUESTIONS["tax"]
        + FRENCH_LAW_QUESTIONS["admin"],
    },
}


OWL_AGENTS_CONFIG_ENV = {
    "development": OWL_AGENTS_DEV,
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
