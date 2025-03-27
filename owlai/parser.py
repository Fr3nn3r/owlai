print("Loading parser module")
from typing import Optional, List, Tuple, Any, Callable
import os
import logging
from langchain.docstore.document import Document as LangchainDocument

from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

from owlai.owlsys import track_time, setup_logging, sprint
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
import pandas as pds
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", category=FutureWarning)

# Get logger using the module name
logger = logging.getLogger(__name__)
