"""
Sistema de busca semântica usando RAG.

Pacote para a implementação de um sistema completo de busca semântica com:
- Ingestão de documentos/arquivos do tipo PDF em banco vetorial
- Busca semântica usando embeddings (OpenAI ou Google) conforme especificação
- Geração de respostas usando LLMs (GPT ou Gemini) conforme especificação
- Interface CLI para a interação com o usuário

Principais componentes:
- config: Gerenciamento e centralização de configurações
- ingest: Processamento e armazenamento de documentos
- search: Busca semântica e geração de respostas
- chat: Interface CLI interativa

Uso:
    >>> from src import Config, answer_questions
    >>> config = Config.from_env()
    >>> answer = answer_questions("Qual o faturamento da empresta X?")
    >>> print(answer)
"""

__version__ = "1.0.0"
__author__ = "Harisson de Freitas"
__license__ = "MIT"

__all__ = [
    "Config",
    "validate_config",
    "get_embeddings",
    "get_llm",
    "answer_questions",
    "search_documents",
    "__version__",
]
from .config import Config, get_embeddings, get_llm, validate_config
from .search import answer_questions, search_documents
