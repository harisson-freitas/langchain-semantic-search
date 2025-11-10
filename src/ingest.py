"""
- Script de ingestão de documentos/arquivos PDF para um banco vetorial
- Upload de um arquivo PDF, dividindo em chunks e armazenando embeddings no banco de dados
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import Config, get_embeddings, validate_config
except ImportError:
    from config import Config, get_embeddings, validate_config


def load_pdf(pdf_path: Path) -> list[Document]:
    """Carrega o arquivo PDF e retorna a lista de documentos.

    Args:
        pdf_path: Caminho para o arquivo PDF

    Returns:
        Lista de documentos carregados do PDF

    Raises:
        FileNotFoundError: Se o arquivo PDF não for encontrado
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado: {pdf_path};\n"
            "Por favor, coloque um arquivo 'document.pdf' no diretório raiz do projeto."
        )

    print(f"\nCarregando PDF: {pdf_path}")
    docs = PyPDFLoader(str(pdf_path)).load()
    print(f"PDF carregado com sucesso: {len(docs)} página(s)")

    return docs


def process_documents(docs: list[Document]) -> tuple[list[Document], list[str]]:
    """Processa os documentos: divide em chunks, limpa metadados e gera IDs.

    Args:
        docs: Lista de documentos originais

    Returns:
        Tupla contendo (documentos processados, IDs dos documentos)
    """
    print("\nDividindo os documentos em chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    )

    splits = splitter.split_documents(docs)

    if not splits:
        raise ValueError("Nenhum chunk gerado. Verifique o conteúdo do arquivo PDF.")

    print(f"{len(splits)} chunks criados")

    print("\nLimpando metadados...")
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    return enriched, ids


def store_in_vectordb(
    documents: list[Document], ids: list[str], config: Config, embeddings
) -> None:
    """Armazena os documentos no banco vetorial.

    Args:
        documents: Lista de documentos processados
        ids: Lista de IDs dos documentos
        config: Configurações da aplicação
        embeddings: Modelo de embeddings a ser utilizado
    """
    print("\nConectando ao banco de dados...")

    assert config.pgvector_collection is not None, (
        "pgvector_collection não pode ser None"
    )
    assert config.pgvector_url is not None, "pgvector_url não pode ser None"

    store = PGVector(
        embeddings=embeddings,
        collection_name=config.pgvector_collection,
        connection=config.pgvector_url,
        use_jsonb=True,
    )

    print(f"\nInserindo {len(documents)} chunks no banco de dados...")
    store.add_documents(documents=documents, ids=ids)


def print_summary(total_chunks: int, collection_name: str) -> None:
    """Imprime o resumo da ingestão.

    Args:
        total_chunks: Número total de chunks armazenados
        collection_name: Nome da coleção no banco
    """
    print("\n" + "=" * 60)
    print("Ingestão concluída com sucesso!")
    print("=" * 60)
    print(f"\nTotal de chunks armazenados: {total_chunks}")
    print(f"Collection: {collection_name}")
    print("\nVocê já pode executar o chat: python src/chat.py")


def main():
    """Função principal para a ingestão."""
    config = Config.from_env()
    validate_config(config)

    current_dir = Path(__file__).parent.parent
    pdf_path = current_dir / "document.pdf"
    docs = load_pdf(pdf_path)

    processed_docs, doc_ids = process_documents(docs)

    print("\nInicializando modelo de embeddings...")
    embeddings = get_embeddings(config)

    store_in_vectordb(processed_docs, doc_ids, config, embeddings)

    assert config.pgvector_collection is not None

    print_summary(len(processed_docs), config.pgvector_collection)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nErro durante a ingestão: {str(e)}")
        raise
