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
MAX_SALARIO = 13000
IDADE_MIN = 16
IDADE_MAX = 70
PROB_FALTA = 0.05  # Probabilidade de campo estar vazio

FEEDBACKS = ["Bom", "Regular", "Ruim"]

def gerar_idade():
    return random.randint(IDADE_MIN, IDADE_MAX)

def gerar_renda():
    return round(random.uniform(MIN_SALARIO, MAX_SALARIO), 2)

def gerar_nota():
    return round(random.uniform(0, 10), 1)

def feedback_aleatorio_complexo(idade, renda, categoria, nota, cidade):
    """
    Gera um feedback sem relação direta com as variáveis.
    Usa um hash pseudoaleatório com base na soma de valores derivados dos campos,
    simulando um comportamento subjetivo e imprevisível.
    """

    # Combinação arbitrária e não linear dos campos
    seed_value = (
        hash((idade or 0)) % 100 +
        hash((renda or 0)) % 100 +
        hash(categoria) % 10 +
        hash((nota or 0)) % 10 +
        hash(cidade) % 100
    )

    # Usamos o valor da semente para escolher o feedback de forma controlada
    random.seed(seed_value)
    feedback = random.choice(FEEDBACKS)
    random.seed()  # Reseta a seed global

    return feedback

def gerar_campo_aleatorio(valor_original, prob_falta=PROB_FALTA):
    """Simula valores faltantes."""
    if random.random() < prob_falta:
        return None
    return valor_original

def gerar_linha(id_registro):
    idade = gerar_idade()
    renda = gerar_renda()
    cidade = random.choice(CAPITAIS)
    categoria = random.choice(CATEGORIAS)
    nota = gerar_nota()

    # Gerar feedback de forma imprevisível e indireta
    feedback = feedback_aleatorio_complexo(idade, renda, categoria, nota, cidade)

    # Simular campos faltando
    idade = gerar_campo_aleatorio(idade)
    renda = gerar_campo_aleatorio(renda)
    cidade = gerar_campo_aleatorio(cidade)
    nota = gerar_campo_aleatorio(nota)

    return [id_registro, idade, renda, cidade, categoria, nota, feedback]

def main():
    parser = argparse.ArgumentParser(description="Gera dataset sintético com feedback não diretamente relacionado aos campos.")
    parser.add_argument("qtd_linhas", type=int, help="Quantidade de linhas do dataset.")
    parser.add_argument("inicio_id", type=int, help="Valor inicial do ID.")

    args = parser.parse_args()

    with open("dataset_feedback_indireto.csv", mode="w", newline='', encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["id", "idade", "renda", "cidade", "categoria", "nota", "feedback"])

        for i in range(args.qtd_linhas):
            id_registro = args.inicio_id + i
            linha = gerar_linha(id_registro)

            # Substituir None por string vazia para o CSV
            linha_limpa = ['' if x is None else x for x in linha]
            writer.writerow(linha_limpa)

    print(f"\n\nDataset gerado com {args.qtd_linhas} registros, iniciando pelo ID {args.inicio_id}.\n")
    print("Feedback foi gerado de forma não diretamente relacionada aos outros campos.\n\n")

if __name__ == "__main__":
    main()