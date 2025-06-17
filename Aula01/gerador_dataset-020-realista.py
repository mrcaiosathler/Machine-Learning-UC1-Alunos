import random
import argparse
import csv

# Parâmetros configuráveis
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
PROBABILIDADE_FALTA = 0.05  # 5% de chance de falta em cada campo opcional

def gerar_idade():
    return random.randint(IDADE_MIN, IDADE_MAX)

def gerar_renda():
    return round(random.uniform(MIN_SALARIO, MAX_SALARIO), 2)

def gerar_nota():
    return round(random.uniform(0, 10), 1)

def determinar_feedback(idade, renda, categoria, nota):
    """Determina feedback com base em múltiplas variáveis."""
    if idade < 18 and renda > 5000:
        return random.choice(['Regular', 'Ruim'])  # Situação improvável
    elif categoria == 'A' and nota >= 7:
        if renda < 2500:
            return 'Bom' if random.random() < 0.7 else 'Regular'
        else:
            return 'Bom'
    elif categoria == 'B' and nota >= 6:
        if renda > 9000:
            return 'Regular'  # Desconforto com alto salário em categoria B
        else:
            return 'Bom' if random.random() < 0.5 else 'Regular'
    elif nota < 4:
        return 'Ruim'
    elif nota < 6:
        if idade > 50:
            return 'Regular'
        else:
            return random.choice(['Regular', 'Ruim'])
    else:
        if renda < 1500:
            return 'Bom' if random.random() < 0.8 else 'Regular'
        else:
            return 'Bom'

def gerar_campo_aleatorio(valor_original, probabilidade_falta=PROBABILIDADE_FALTA):
    """Simula valores faltantes."""
    if random.random() < probabilidade_falta:
        return None  # Representa um valor faltante
    return valor_original

def gerar_linha(id_registro):
    idade = gerar_idade()
    renda = gerar_renda()
    cidade = random.choice(CAPITAIS)
    categoria = random.choice(CATEGORIAS)
    nota = gerar_nota()

    feedback = determinar_feedback(idade, renda, categoria, nota)

    # Simular campos faltando
    idade = gerar_campo_aleatorio(idade)
    renda = gerar_campo_aleatorio(renda)
    cidade = gerar_campo_aleatorio(cidade)
    nota = gerar_campo_aleatorio(nota)

    return [id_registro, idade, renda, cidade, categoria, nota, feedback]

def main():
    parser = argparse.ArgumentParser(description="Gera um dataset sintetico com informacoes realistas.")
    parser.add_argument("qtd_linhas", type=int, help="Quantidade de linhas do dataset.")
    parser.add_argument("inicio_id", type=int, help="Valor inicial do ID.")

    args = parser.parse_args()

    with open("dataset_feedback_realista.csv", mode="w", newline='', encoding="utf-8") as arquivo:
        writer = csv.writer(arquivo)
        writer.writerow(["id", "idade", "renda", "cidade", "categoria", "nota", "feedback"])

        for i in range(args.qtd_linhas):
            id_registro = args.inicio_id + i
            linha = gerar_linha(id_registro)

            # Substituir None por string vazia na saída CSV
            linha_limpa = ['' if x is None else x for x in linha]
            writer.writerow(linha_limpa)

    print(f"\n\nDataset realista gerado com {args.qtd_linhas} registros, iniciando pelo ID {args.inicio_id}.\n\n")

if __name__ == "__main__":
    main()