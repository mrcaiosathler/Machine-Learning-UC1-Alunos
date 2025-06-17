import random
import argparse
import csv

# Configurações básicas
CAPITAIS = [
    "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Brasilia",
    "Curitiba", "Recife", "Porto Alegre", "Manaus", "Florianopolis",
    "Fortaleza", "Natal", "Joao Pessoa", "Maceio", "Campo Grande",
    "Cuiaba", "Goiania", "Teresina", "Palmas", "Aracaju"
]

CATEGORIAS = ['A', 'B']
FEEDBACKS = ['Bom', 'Regular', 'Ruim']

IDADE_MIN = 16
IDADE_MAX = 70
RENDA_MIN = 1300   # 1 salário mínimo
RENDA_MAX = 13000  # 10 salários mínimos

PROBABILIDADE_FALTA = 0.05  # 5% de chance de campo estar vazio

def gerar_valor_ou_falta(valor):
    if random.random() < PROBABILIDADE_FALTA:
        return ''
    return valor

def main():
    parser = argparse.ArgumentParser(description="Gera um dataset simples com informações aleatórias.")
    parser.add_argument("qtd_linhas", type=int, help="Quantidade de linhas do dataset.")
    parser.add_argument("inicio_id", type=int, help="Valor inicial do ID.")

    args = parser.parse_args()

    with open("dataset_feedbak_simples.csv", "w", newline="", encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["id", "idade", "renda", "cidade", "categoria", "nota", "feedback"])

        for i in range(args.qtd_linhas):
            id_registro = args.inicio_id + i
            idade = random.randint(IDADE_MIN, IDADE_MAX)
            renda = round(random.uniform(RENDA_MIN, RENDA_MAX), 2)
            cidade = random.choice(CAPITAIS)
            categoria = random.choice(CATEGORIAS)
            nota = round(random.uniform(0, 10), 1)
            feedback = random.choice(FEEDBACKS)

            # Aplicar valores faltantes
            idade = gerar_valor_ou_falta(idade)
            renda = gerar_valor_ou_falta(renda)
            cidade = gerar_valor_ou_falta(cidade)
            nota = gerar_valor_ou_falta(nota)

            writer.writerow([id_registro, idade, renda, cidade, categoria, nota, feedback])

    print(f"\n\nDataset simples gerado com {args.qtd_linhas} registros, iniciando pelo ID {args.inicio_id}.\n\n")

if __name__ == "__main__":
    main()