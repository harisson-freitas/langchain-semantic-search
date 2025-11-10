"""
- Módulo centralizado contendo configurações e validações de variáveis de ambiente
- Gerenciamento das configurações de AI providers (OpenAI/Google) e banco vetorial
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


@dataclass
class Config:
    """Configurações da aplicação carregadas das variáveis de ambiente."""

    ai_provider: str
    pgvector_url: str | None
    pgvector_collection: str | None
    openai_api_key: str | None
    openai_embedding_model: str | None
    openai_llm_model: str | None
    google_api_key: str | None
    google_embedding_model: str | None
    google_llm_model: str | None

    @classmethod
    def from_env(cls) -> "Config":
        """Carrega configurações das variáveis de ambiente."""
        ai_provider = os.getenv("AI_PROVIDER", "openai").lower()

        return cls(
            ai_provider=ai_provider,
            pgvector_url=os.getenv("PGVECTOR_URL"),
            pgvector_collection=os.getenv("PGVECTOR_COLLECTION"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL"),
            openai_llm_model=os.getenv("OPENAI_LLM_MODEL"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
            google_llm_model=os.getenv("GOOGLE_LLM_MODEL"),
        )


def validate_config(config: Config) -> None:
    """Valida se todas as variáveis de ambiente necessárias foram configuradas."""
    print("=" * 60)
    print("Verificando variáveis de ambiente...")
    print("=" * 60)

    print(f"\nProvider de AI: {config.ai_provider}")

    if config.ai_provider not in ["openai", "google"]:
        raise ValueError(
            f"AI_PROVIDER inválido: {config.ai_provider}. Use 'openai' ou 'google'"
        )

    if not config.pgvector_url:
        raise RuntimeError("Variável de ambiente PGVECTOR_URL não está definida")
    print("PGVECTOR_URL: Definida")

    if not config.pgvector_collection:
        raise RuntimeError("Variável de ambiente PGVECTOR_COLLECTION não está definida")
    print("PGVECTOR_COLLECTION: Definida")

    if config.ai_provider == "openai":
        if not config.openai_api_key:
            raise RuntimeError("Variável de ambiente OPENAI_API_KEY não está definida")
        print("OPENAI_API_KEY: Definida")

        if not config.openai_embedding_model:
            raise RuntimeError(
                "Variável de ambiente OPENAI_EMBEDDING_MODEL não está definida"
            )
        print("OPENAI_EMBEDDING_MODEL: Definida")

        if not config.openai_llm_model:
            raise RuntimeError(
                "Variável de ambiente OPENAI_LLM_MODEL não está definida"
            )
        print("OPENAI_LLM_MODEL: Definida")
    elif config.ai_provider == "google":
        if not config.google_api_key:
            raise RuntimeError("Variável de ambiente GOOGLE_API_KEY não está definida")
        print("GOOGLE_API_KEY: Definida")

        if not config.google_embedding_model:
            raise RuntimeError(
                "Variável de ambiente GOOGLE_EMBEDDING_MODEL não está definida"
            )
        print("GOOGLE_EMBEDDING_MODEL: Definida")

        if not config.google_llm_model:
            raise RuntimeError(
                "Variável de ambiente GOOGLE_LLM_MODEL não está definida"
            )
        print("GOOGLE_LLM_MODEL: Definida")

    print("\n" + "=" * 60)


def get_embeddings(config: Config):
    """Retorna o modelo de embeddings baseado no provedor configurado."""
    if config.ai_provider == "openai":
        assert config.openai_embedding_model is not None, (
            "openai_embedding_model não pode ser None"
        )
        print(f"Usando OpenAI embeddings: {config.openai_embedding_model}")
        return OpenAIEmbeddings(model=config.openai_embedding_model)
    elif config.ai_provider == "google":
        assert config.google_embedding_model is not None, (
            "google_embedding_model não pode ser None"
        )
        print(
            f"Usando Google Generative AI embeddings: {config.google_embedding_model}"
        )
        return GoogleGenerativeAIEmbeddings(model=config.google_embedding_model)
    else:
        raise ValueError(f"AI_PROVIDER inválido: {config.ai_provider}")


def get_llm(config: Config) -> BaseChatModel:
    """Retorna o modelo LLM baseado no provider configurado."""
    if config.ai_provider == "openai":
        assert config.openai_llm_model is not None, "openai_llm_model não pode ser None"
        print(f"Usando OpenAI LLM: {config.openai_llm_model}")
        return ChatOpenAI(model=config.openai_llm_model, temperature=0)
    elif config.ai_provider == "google":
        assert config.google_llm_model is not None, "google_llm_model não pode ser None"
        print(f"Usando Google Generative AI LLM: {config.google_llm_model}")
        return ChatGoogleGenerativeAI(model=config.google_llm_model, temperature=0)
    else:
        raise ValueError(f"AI_PROVIDER inválido: {config.ai_provider}")
