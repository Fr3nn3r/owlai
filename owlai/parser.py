print("Loading parser module")
from typing import Optional, List, Tuple, Any, Callable
import os
import logging
from langchain.docstore.document import Document as LangchainDocument

from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

from owlai.owlsys import track_time
import warnings
from tqdm import tqdm

import fitz
import re
from fitz import Document
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from owlai.owlsys import track_time

from tqdm import tqdm

from typing import List
import os
import logging
import traceback
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

from owlai.owlsys import load_logger_config, sprint

logger = logging.getLogger("main")
