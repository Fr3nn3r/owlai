if __name__ == "__main__":
    import time  # this prevents reloading when multi processing is used
    from datetime import datetime

    main_start_time = datetime.now()

    import os
    import transformers

    import logging
    import logging.config
    import yaml
    import json

    from owlai.rag import LocalRAGTool
    from owlai.db import TOOLS_CONFIG, PROMPT_CONFIG
    from langchain.chat_models import init_chat_model

    from tqdm import tqdm

    from owlai.owlsys import get_system_info

    def load_logger_config():
        with open("logging.yaml", "r") as logger_config:
            config = yaml.safe_load(logger_config)
            logging.config.dictConfig(config)

    transformers.logging.set_verbosity_error()

    load_logger_config()

    logger = logging.getLogger("ragtool")

    def main():

        execution_log = []

        now = datetime.now()
        start_time = main_start_time
        start_date_time = now.strftime("%Y-%m-%d-%Hh%Mm%f")[:-3]
        execution_log.append({"Start time": str(start_date_time)})
        execution_log.append({"Application started": str(start_time - now)})
        start_time = now

        datasets = [
            {
                "input_data_folder": "data/dataset-0000",
                "questions": [
                    "Who is Rich Draves?",
                    "What happened to Paul Graham in the fall of 1992?",
                    "What was the last result of the AC Milan soccer team?",
                    "What did Paul Graham do growing up?",
                    "What did Paul Graham do during his school days?",
                    "What languages did Paul Graham use?",
                    "How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?",
                    "How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?",
                    "What is the color of henry the fourth white horse?",
                ],
            },
            {
                "input_data_folder": "data/dataset-0001",
                "questions": [
                    "Who is Naruto?",
                    "Provide details about Orochimaru",
                    "Who is the strongest ninja in the world?",
                    "How is sasuke's personality",
                    "Who is the sensei of naruto?",
                    "What is a sharingan?",
                    "What is the akatsuki?",
                    "Who is the first Hokage?",
                    "What was the last result of the AC Milan soccer team?",
                    "What is the color of henry the fourth white horse?",
                ],
            },
            {
                "input_data_folder": "data/dataset-0002",
                "questions": [
                    "Quelles sont les différences essentielles entre la responsabilité contractuelle et délictuelle ?",
                    "Expliquez les conditions de validité d'un contrat et les conséquences juridiques de leur non-respect.",
                    "Comment s'articule le mécanisme de la prescription en matière civile ? Citez des exemples de délais spécifiques.",
                    "Quelles sont les principales différences entre une SARL et une SAS en termes de gouvernance et de responsabilité ?",
                    "Expliquez le régime juridique des actes de concurrence déloyale et leurs sanctions.",
                    "Comment s'opère la cession d'un fonds de commerce et quelles sont les formalités obligatoires ?",
                    "Quelles sont les modalités légales de rupture d'un CDI et leurs spécificités procédurales ?",
                    "Expliquez les obligations de l'employeur en matière de santé et sécurité au travail.",
                    "Comment caractériser juridiquement le harcèlement moral et quelles sont les voies de recours pour un salarié ?",
                    "Quelles sont les conditions de validité d'un bail commercial et ses clauses essentielles ?",
                    "Expliquez les différentes servitudes et leurs modes d'établissement.",
                    "Comment s'opère la vente en l'état futur d'achèvement (VEFA) et quelles garanties offre-t-elle ?",
                    "Un client vous consulte pour un litige avec son fournisseur. Quelles informations collecteriez-vous et quelle serait votre méthodologie d'analyse ?",
                    "Comment rédigeriez-vous une clause de non-concurrence qui soit à la fois protectrice pour l'entreprise et juridiquement valable ?",
                    "Face à un conflit d'intérêts potentiel, quelle démarche adopteriez-vous ?",
                    "Explique la gestion en france de la confusion des peines",
                ],
            },
            {
                "input_data_folder": "data/dataset-0003",
                "questions": [
                    "What is Arakis?",
                    "Who is Duncan Idaho?",
                    "Who is the traitor?",
                    "What are the powers of the Bene Gesserit?",
                    "How does Paul defeat the Emperor?",
                ],
            },
            {
                "input_data_folder": "data/dataset-0004",  # dataset 4 droit fiscal
                "questions": [
                    "Quelles sont les principales obligations fiscales d’une entreprise soumise à l’impôt sur les sociétés (IS) en France ?",
                    "Quels sont les critères permettant de déterminer si une opération est soumise à la TVA en France ?",
                    "Quelles sont les principales conventions fiscales internationales et comment permettent-elles d’éviter la double imposition ?",
                    "Quels sont les principaux droits et obligations d’une entreprise lors d’un contrôle fiscal ?",
                    "Quels sont les recours possibles pour une entreprise contestant un redressement fiscal ?",
                    "Quels sont les principaux impôts applicables aux transmissions de patrimoine en France ?",
                    "Quelles sont les obligations déclaratives en matière de prix de transfert pour les entreprises multinationales ?",
                    "Comment la notion d’abus de droit fiscal est-elle définie en droit français et quelles en sont les conséquences ?",
                    "Quels sont les principaux enjeux de la conformité fiscale pour une entreprise et comment un juriste fiscaliste peut-il y contribuer ?",
                    "Pouvez-vous nous parler d’une récente réforme fiscale qui a eu un impact significatif sur les entreprises en France ?",
                ],
            },
        ]

        dataset = datasets[4]

        # Override the default configuration
        TOOLS_CONFIG["owl_memory_tool"]["input_data_folders"] = []
        TOOLS_CONFIG["owl_memory_tool"]["model_name"] = "mistral-large-latest"
        TOOLS_CONFIG["owl_memory_tool"]["model_provider"] = "mistralai"
        TOOLS_CONFIG["owl_memory_tool"]["system_prompt"] = PROMPT_CONFIG["rag-fr-v0"]
        TOOLS_CONFIG["owl_memory_tool"]["control_llm"] = "gpt-4o"

        rag_tool = LocalRAGTool(**TOOLS_CONFIG["owl_memory_tool"])

        embedding_model = rag_tool._embeddings

        input_data_folder = dataset["input_data_folder"]

        KNOWLEDGE_VECTOR_DATABASE = rag_tool.load_or_create_vector_store(
            input_data_folder, embedding_model
        )

        now = datetime.now()
        execution_log.append({"Vector store loaded": str(start_time - now)})
        start_time = now

        rag_tool._vector_stores = KNOWLEDGE_VECTOR_DATABASE

        questions = dataset["questions"]

        qa_results = []
        qa_results.append({"system_info": get_system_info()})

        qa_results.append({"test_parameters": TOOLS_CONFIG["owl_memory_tool"]})

        qa_results.append({"dataset": dataset})

        test_report_filename = f"{start_date_time}-qa_results.json"

        gpt = init_chat_model("gpt-4o", temperature=0.1, max_tokens=4096)

        for i, question in enumerate(questions, 1):

            answer = rag_tool.rag_question(question)
            logger.info(f"USER QUERY : {question}")
            logger.info(f"ANSWER : {answer}")

            now = datetime.now()
            answer_time = now - start_time
            start_time = now

            answer_control_llm = gpt.invoke(question).content

            now = datetime.now()
            answer_time_control_llm = now - start_time
            start_time = now

            qa_results.append(
                {
                    f"Test #{i}": {
                        "question": question,
                        "answer": answer,
                        "answer_control_llm": answer_control_llm,
                        "answer_time": str(answer_time),
                        "answer_time_control_llm": str(answer_time_control_llm),
                    }
                }
            )

        now = datetime.now()
        execution_log.append({"Questions answered": str(start_time - now)})
        execution_log.append({"Total time": str(now - main_start_time)})

        qa_results.append({"execution_log": execution_log})

        # Save results to JSON file
        output_file = os.path.join(input_data_folder, test_report_filename)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
        # logger.info(f"SOURCE DOCS : {relevant_docs}")

    print("Application started")
    from multiprocessing import freeze_support
    from dotenv import load_dotenv

    load_dotenv()
    freeze_support()
    main()
