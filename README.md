# Ingestão e busca semântica com LangChain e Postgres

Sistema de busca semântica utilizando RAG (Retrieval-Augmented Generation) com LangChain, PostgreSQL + pgVector para responder perguntas baseadas em documentos PDF.

## Tecnologias

- **Python 3.13+**
- **LangChain** - Framework para aplicações com LLM
- **PostgreSQL + pgVector** - Banco de dados vetorial
- **OpenAI / Google Gemini** - Modelos de embeddings e LLM
- **Docker & Docker Compose** - Containerização

## Pré-requisitos

Antes de começar, certifique-se de ter instalado:

### Software necessário

- **Python 3.13 ou superior**
  - Verifique a versão: `python3 --version`
  - Download: [python.org](https://www.python.org/downloads/)

- **pip** (gerenciador de pacotes Python)
  - Geralmente já vem com Python
  - Verifique: `pip --version`

- **Docker** e **Docker Compose**
  - Necessário para rodar o PostgreSQL + pgVector
  - Docker Desktop inclui o Docker Compose
  - Download: [docker.com](https://www.docker.com/get-started)
  - Verifique: `docker --version` e `docker compose version`

### Recursos do sistema

- **Porta 5432** disponível (usada pelo PostgreSQL)
- **Espaço em disco**: ~500MB mínimo (banco de dados + imagens Docker)
- **Memória RAM**: 4GB recomendado

### API Keys

Você precisará de uma conta e API key em **um** dos provedores:

- **OpenAI**
  - Cadastro: [platform.openai.com](https://platform.openai.com)
  - API Keys: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

- **Google Gemini**
  - Cadastro: [ai.google.dev](https://ai.google.dev)
  - API Keys: [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

## 🔧 Configuração

### 1. Clone o repositório

```bash
git clone <seu-repositorio>
cd langchain-semantic-search
```

### 2. Crie e ative o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Copie o arquivo de exemplo e configure suas credenciais:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione sua API Key:

**Para OpenAI:**
```bash
AI_PROVIDER=openai
OPENAI_API_KEY=sua_chave_aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_LLM_MODEL=gpt-5-nano
```

**Para Google Gemini:**
```bash
AI_PROVIDER=google
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_LLM_MODEL=gemini-2.5-flash-lite
```

### 5. Adicione seu documento PDF

Coloque o arquivo PDF que deseja processar na raiz do projeto com o nome `document.pdf`

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde alguns segundos para o banco inicializar completamente.

### 2. Executar ingestão do PDF

```bash
python src/ingest.py
```

Este comando irá:
- Carregar o PDF
- Dividir em chunks de 1000 caracteres
- Gerar embeddings
- Armazenar no banco vetorial

### 3. Rodar o chat

```bash
python src/chat.py
```

## Exemplo de uso

```
============================================================
Sistema de Busca Semântica - Chat
Digite sua pergunta ou 'sair' para encerrar.
============================================================

Faça sua pergunta: Qual o faturamento da Empresa SuperTechIABrazil?

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?

Processando...

RESPOSTA: O faturamento foi de 10 milhões de reais.

------------------------------------------------------------

Faça sua pergunta: Quantos clientes temos em 2024?

PERGUNTA: Quantos clientes temos em 2024?

Processando...

RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Parar o banco de dados

```bash
docker compose down
```

Para remover também os dados:
```bash
docker compose down -v
```

## Estrutura do projeto

```
.
├── docker-compose.yml          # Configuração do PostgreSQL + pgVector
├── requirements.txt            # Dependências Python
├── .env.example               # Template de variáveis de ambiente
├── .env                       # Variáveis de ambiente (não versionado)
├── document.pdf               # PDF para ingestão
├── src/
│   ├── __init__.py           # Inicialização do módulo
│   ├── config.py             # Configurações e validações
│   ├── ingest.py             # Script de ingestão do PDF
│   ├── search.py             # Funções de busca semântica
│   └── chat.py               # Interface CLI
└── README.md                  # Este arquivo
```

## Como funciona

1. **Ingestão**: O PDF é carregado e dividido em chunks de 1000 caracteres com overlap de 150
2. **Embeddings**: Cada chunk é transformado em vetor usando modelo de embeddings
3. **Armazenamento**: Vetores são salvos no PostgreSQL com extensão pgVector
4. **Busca**: Pergunta do usuário é vetorizada e comparada com os chunks (k=10)
5. **Resposta**: Os chunks mais relevantes são enviados como contexto para a LLM gerar a resposta

## Regras de resposta

O sistema responde **apenas** com base no conteúdo do PDF:
- Se a informação estiver no documento, retorna a resposta
- Se não estiver, retorna: "Não tenho informações necessárias para responder sua pergunta."
- Nunca inventa ou usa conhecimento externo

## Dependências principais

- `langchain` - Framework principal
- `langchain-openai` - Integração OpenAI
- `langchain-google-genai` - Integração Google
- `langchain-postgres` - Integração PostgreSQL
- `langchain-community` - Loaders e utilitários
- `pypdf` - Leitura de PDFs
- `pgvector` - Extensão vetorial para PostgreSQL

## Troubleshooting

**Erro ao conectar no banco:**
- Verifique se o Docker está rodando
- Confirme que a porta 5432 não está em uso
- Aguarde alguns segundos após `docker compose up`

**Erro de API Key:**
- Verifique se a chave está correta no arquivo `.env`
- Confirme que não há espaços extras
- Para OpenAI: https://platform.openai.com/api-keys
- Para Google: https://makersuite.google.com/app/apikey

**Erro no import:**
- Ative o ambiente virtual: `source venv/bin/activate`
- Reinstale as dependências: `pip install -r requirements.txt`

## Licença

Este projeto foi desenvolvido para fins educacionais.
