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

    from owlai.owlsys import get_system_info, track_time

    import fitz  # PyMuPDF
    import re

    def load_logger_config():
        with open("logging.yaml", "r") as logger_config:
            config = yaml.safe_load(logger_config)
            logging.config.dictConfig(config)

    transformers.logging.set_verbosity_error()

    load_logger_config()

    logger = logging.getLogger("ragtool")

    def extract_footer_fr_law(pdf_path):
        doc = fitz.open(pdf_path)
        footers = []

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Extract text and split into lines
            text = page.get_text("text")
            lines = text.split("\n")

            if len(lines) > 1:
                footer = lines[-2:]  # Assume footer is in the last two lines
                footers.append((page_num + 1, " | ".join(footer)))

        return footers

    def extract_metadata_fr_law(pdf_path):
        """
        Extract metadata from a PDF file footer (french law from legifrance.gouv.fr).

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            dict: Dictionary containing title, last_modification, and doc_generated_on
        """
        # Get footer from the first page
        footers = extract_footer_fr_law(pdf_path)
        if not footers:
            return {"title": "", "last_modification": "", "doc_generated_on": ""}

        footer = footers[0][1]

        # Regular Expression to Extract Components
        match = re.match(r"^(.*?)\s*-\s*(.*?)\s*-\s*(.*?)$", footer)
        if match:
            title = match.group(1).strip()
            last_modification = match.group(2).strip()
            doc_generated_on = match.group(3).strip()

            return {
                "title": title,
                "last_modification": last_modification,
                "doc_generated_on": doc_generated_on,
            }

        return {"title": "", "last_modification": "", "doc_generated_on": ""}

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
                "input_data_folder": "data/dataset-0002",
                "input_vector_store": "data/dataset-0002/vector_db",
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
                "input_data_folder": "data/dataset-0004",  # dataset 4 droit fiscal
                "input_vector_store": "data/dataset-0004/vector_db",
                "questions": [
                    "Quelles sont les principales obligations fiscales d'une entreprise soumise à l'impôt sur les sociétés (IS) en France ?",
                    "Quels sont les critères permettant de déterminer si une opération est soumise à la TVA en France ?",
                    "Quelles sont les principales conventions fiscales internationales et comment permettent-elles d'éviter la double imposition ?",
                    "Quels sont les principaux droits et obligations d'une entreprise lors d'un contrôle fiscal ?",
                    "Quels sont les recours possibles pour une entreprise contestant un redressement fiscal ?",
                    "Quels sont les principaux impôts applicables aux transmissions de patrimoine en France ?",
                    "Quelles sont les obligations déclaratives en matière de prix de transfert pour les entreprises multinationales ?",
                    "Comment la notion d'abus de droit fiscal est-elle définie en droit français et quelles en sont les conséquences ?",
                    "Quels sont les principaux enjeux de la conformité fiscale pour une entreprise et comment un juriste fiscaliste peut-il y contribuer ?",
                    "Pouvez-vous nous parler d'une récente réforme fiscale qui a eu un impact significatif sur les entreprises en France ?",
                ],
            },
            {
                "input_data_folder": "data/dataset-0005",
                "input_vector_store": "data/dataset-0005/vector_db",
                "questions": [
                    "Quelles sont les principales obligations fiscales d'une entreprise soumise à l'impôt sur les sociétés (IS) en France ?",
                    "Quels sont les critères permettant de déterminer si une opération est soumise à la TVA en France ?",
                    "Quelles sont les principales conventions fiscales internationales et comment permettent-elles d'éviter la double imposition ?",
                    "Quels sont les principaux droits et obligations d'une entreprise lors d'un contrôle fiscal ?",
                    "Quels sont les recours possibles pour une entreprise contestant un redressement fiscal ?",
                    "Quels sont les principaux impôts applicables aux transmissions de patrimoine en France ?",
                    "Quelles sont les obligations déclaratives en matière de prix de transfert pour les entreprises multinationales ?",
                    "Comment la notion d'abus de droit fiscal est-elle définie en droit français et quelles en sont les conséquences ?",
                    "Quels sont les principaux enjeux de la conformité fiscale pour une entreprise et comment un juriste fiscaliste peut-il y contribuer ?",
                    "Pouvez-vous nous parler d'une récente réforme fiscale qui a eu un impact significatif sur les entreprises en France ?",
                ],
            },
        ]

        dataset = datasets[2]

        # Override the default configuration
        TOOLS_CONFIG["owl_memory_tool"]["input_data_folders"] = []
        TOOLS_CONFIG["owl_memory_tool"]["model_name"] = "mistral-large-latest"
        TOOLS_CONFIG["owl_memory_tool"]["model_provider"] = "mistralai"
        TOOLS_CONFIG["owl_memory_tool"]["system_prompt"] = PROMPT_CONFIG["rag-fr-v0"]
        TOOLS_CONFIG["owl_memory_tool"]["control_llm"] = "gpt-4o"

        rag_tool = LocalRAGTool(**TOOLS_CONFIG["owl_memory_tool"])

        embedding_model = rag_tool._embeddings

        input_data_folder = dataset["input_data_folder"]

        # Using the metadata extractor function with the vector store creation
        # this is allowing the caller to specify how the metadata is extracted from the documents
        # and stored in the vector store
        with track_time("Vector store loading", execution_log):
            KNOWLEDGE_VECTOR_DATABASE = rag_tool.load_or_create_vector_store(
                input_data_folder,
                embedding_model,
                metadata_extractor=extract_metadata_fr_law,
            )

        rag_tool._vector_stores = KNOWLEDGE_VECTOR_DATABASE

        questions = dataset["questions"]

        qa_results = []
        qa_results.append({"system_info": get_system_info()})

        qa_results.append({"test_parameters": TOOLS_CONFIG["owl_memory_tool"]})

        qa_results.append({"dataset": dataset})

        test_report_filename = f"{start_date_time}-qa_results.json"

        gpt = init_chat_model("gpt-4o", temperature=0.1, max_tokens=4096)

        for i, question in enumerate(questions, 1):

            with track_time(f"RAG Question {i}", execution_log):
                answer = rag_tool.rag_question(question)
                logger.info(f"USER QUERY : {question}")
                logger.info(f"ANSWER : {answer}")

            with track_time(f"Control LLM Question {i}", execution_log):
                answer_control_llm = gpt.invoke(question).content

            qa_results.append(
                {
                    f"Test #{i}": {
                        "question": question,
                        "answer": answer,
                        "answer_control_llm": answer_control_llm,
                    }
                }
            )

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
