"""
- Módulo para realizar busca semântica no banco vetorial
- Fornece funções de busca de documentos e geração de respostas usando LLM
"""

import sys
from pathlib import Path

from langchain_postgres import PGVector

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from .config import Config, get_embeddings, get_llm, validate_config
except ImportError:
    from config import Config, get_embeddings, get_llm, validate_config


def get_vector_store(config: Config) -> PGVector:
    """Retona a instância do vector conectado ao banco"""

    assert config.pgvector_collection is not None, (
        "pgvector_collection não pode ser None"
    )
    assert config.pgvector_url is not None, "pgvector_url não pode ser None"

    store = PGVector(
        embeddings=get_embeddings(config),
        collection_name=config.pgvector_collection,
        connection=config.pgvector_url,
        use_jsonb=True,
    )
    return store


def search_documents(config: Config, query: str, k: int = 10):
    """
    Busca documentos relevantes no banco vetorial.

    Args:
        query: Pergunta do usuário
        k: Número de resultados a retornar (padrão: 10)
    Returns:
        Lista de tuplas (documento, score)
    """
    store = get_vector_store(config)
    results = store.similarity_search_with_score(query, k=k)
    return results


def build_prompt(query: str, context: str) -> str:
    """
    Constrói o prompt para a LLM conforme especificação do trabalho.

    Args:
        query: Pergunta do usuário
        context: Contexto extraído dos documentos relevantes

    Returns:
        Prompt formatado
    """
    prompt = f"""CONTEXTO:
    {context}

    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se a informação não estiver explicitamente no CONTEXTO, responda:
     "Não tenho informações necessárias para responder sua pergunta."
    - Nunca invente ou use conhecimento externo.
    - Nunca produza opiniões ou interpretações além do que está escrito.

    EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
    Pergunta: "Qual é a capital da França?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Quantos clientes temos em 2024?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Você acha isso bom ou ruim?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    PERGUNTA DO USUÁRIO:
    {query}

    RESPONDA A "PERGUNTA DO USUÁRIO":"""

    return prompt


def answer_questions(query: str, k: int = 10, verbose: bool = False) -> str:
    """
    Responde uma pergunta usando RAG (Retrieval-Augmented Generation).

    Args:
        query: Pergunta do usuário
        k: Número de documentos a buscar (padrão: 10)
        verbose: Se True, exibe informações de debug

    Returns:
        Resposta gerada pela LLM
    """

    config = Config.from_env()
    validate_config(config)

    if verbose:
        print(f"\nBuscando {k} documentos relevantes...")

    results = search_documents(config, query, k=k)

    if verbose:
        print(f"Encontrados {len(results)} documentos")
        print("\nScores de relevância:")
        for i, (_, score) in enumerate(results[:3], 1):
            print(f"  {i}. Score: {score:.4f}")

    context = "\n\n".join([doc.page_content for doc, _ in results])

    if verbose:
        print(f"\nTamanho do contexto: {len(context)} caracteres")

    prompt = build_prompt(query, context)

    if verbose:
        print("\nGerando resposta...")

    llm = get_llm(config)
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        content = response.content

        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return "".join(str(item) for item in content)
        else:
            return str(content)
    else:
        return str(response)


if __name__ == "__main__":
    question = "Qual o faturamento da empresa?"
    print(f"\nPERGUNTA: {question}\n")

    answer = answer_questions(question, verbose=True)
    print(f"\nRESPOSTA: {answer}\n")
