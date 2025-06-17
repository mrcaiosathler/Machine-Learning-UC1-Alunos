import random
import argparse
import csv

# Configurações
CAPITAIS = [
    "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Brasilia",
    "Curitiba", "Recife", "Porto Alegre", "Manaus", "Florianopolis",
    "Fortaleza", "Natal", "Joao Pessoa", "Maceio", "Campo Grande",
    "Cuiaba", "Goiania", "Teresina", "Palmas", "Aracaju"
]

CATEGORIAS = ['A', 'B']
MIN_SALARIO = 1300
MAX_SALARIO = 13000  # 10 salarios minimos aproximadamente

def gerar_idade():
    return random.randint(16, 70)

def gerar_renda():
    return round(random.uniform(MIN_SALARIO, MAX_SALARIO), 2)

def gerar_nota():
    return round(random.uniform(0, 10), 1)

def determinar_feedback(nota, renda, categoria):
    # Regras simples e aleatorias para diversificar o feedback
    if nota >= 7.5:
        if renda < 3000 and categoria == 'A':
            return 'Bom'
        elif renda > 10000:
            return random.choice(['Regular', 'Bom'])
        else:
            return 'Bom'
    elif 4 <= nota < 7.5:
        if renda > 8000 or categoria == 'B':
            return 'Regular'
        else:
            return 'Bom' if random.random() < 0.3 else 'Regular'
    else:
        if renda < 2000:
            return 'Ruim'
        else:
            return random.choice(['Ruim', 'Regular'])

def gerar_linha(id_registro):
    idade = gerar_idade()
    renda = gerar_renda()
    cidade = random.choice(CAPITAIS)
    categoria = random.choice(CATEGORIAS)
    nota = gerar_nota()
    feedback = determinar_feedback(nota, renda, categoria)

    return [id_registro, idade, renda, cidade, categoria, nota, feedback]

def main():
    parser = argparse.ArgumentParser(description="Gera um dataset sintetico com informacoes de usuarios.")
    parser.add_argument("qtd_linhas", type=int, help="Quantidade de linhas do dataset.")
    parser.add_argument("inicio_id", type=int, help="Valor inicial do ID.")

    args = parser.parse_args()

    with open("dataset_feedbak_relacionado.csv", mode="w", newline='', encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["id", "idade", "renda", "cidade", "categoria", "nota", "feedback"])

        for i in range(args.qtd_linhas):
            id_registro = args.inicio_id + i
            linha = gerar_linha(id_registro)
            writer.writerow(linha)

    print(f"\n\nDataset gerado com {args.qtd_linhas} registros, iniciando pelo ID {args.inicio_id}.\n\n")

if __name__ == "__main__":
    main()
