"""
- Interface CLI para que o usuário interaja com o sistema de busca semântica
- Permite ao usuário fazer perguntar e receber respostas com base no conteúdo do arquivo PDF
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from .config import Config, validate_config
    from .search import answer_questions
except ImportError:
    from config import Config, validate_config
    from search import answer_questions


def print_header():
    print("=" * 60)
    print("Sistema de Busca Semântica - Chat")
    print("Digite sua pergunta ou 'sair' para encerrar.")
    print("=" * 60)


def main():
    config = Config.from_env()
    validate_config(config)

    print_header()

    while True:
        try:
            print("-" * 60)
            user_input = input("\nFaça sua pergunta: ").strip()

            if user_input.lower() in ["sair", "exit", "quit", "q"]:
                print("Encerrando o programa. Até mais!")
                break
            if not user_input:
                print("Por favor, digite uma pergunta válida.")
                continue

            print(f"\nPERGUNTA: {user_input}")
            print("\n Processando...")

            answer = answer_questions(user_input, k=10, verbose=False)

            print(f"\nRESPOSTA: {answer}")
            print()

        except KeyboardInterrupt:
            print("Chat interrompido. Até logo!")
            break
        except Exception as e:
            print(f"\nErro ao processar pergunta: {str(e)}")
            print("Tente novamente ou verifique a configuração.\n")


if __name__ == "__main__":
    main()
