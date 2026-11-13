import os
import time
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURAÇÃO DOS FICHEIROS
# ==========================================
NOME_COM_FILTRO = "deter-amz-daily-10-11-2026-14_53_52.csv"
NOME_SEM_FILTRO = "deter-amz-daily-10-11-2026-14_51_43.csv"
SEMENTE = 42


def carregar_planilha(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Ficheiro '{caminho}' não encontrado na pasta!")

    df = pd.read_csv(caminho, encoding="utf-8-sig")

    achadas = [c for c in df.columns if str(c).strip().lower() == "areamunkm"]
    if not achadas:
        raise ValueError(f"Coluna 'areaMunKm' não encontrada em '{caminho}'.")
    coluna = achadas[0]

    df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    df = df.dropna(subset=[coluna]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"'{caminho}' não tem valores válidos em areaMunKm.")

    return df, coluna


# ==========================================
# 2. ALGORITMOS DE ORDENAÇÃO
# ==========================================
def bubble_sort(dados, col):
    dados = dados.copy()
    n = len(dados)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if dados[j][col] > dados[j + 1][col]:
                dados[j], dados[j + 1] = dados[j + 1], dados[j]
    return dados


def selection_sort(dados, col):
    dados = dados.copy()
    n = len(dados)
    for i in range(n - 1):
        menor = i
        for j in range(i + 1, n):
            if dados[j][col] < dados[menor][col]:
                menor = j
        dados[i], dados[menor] = dados[menor], dados[i]
    return dados


def insertion_sort(dados, col):
    dados = dados.copy()
    for i in range(1, len(dados)):
        atual = dados[i]
        j = i - 1
        while j >= 0 and dados[j][col] > atual[col]:
            dados[j + 1] = dados[j]
            j -= 1
        dados[j + 1] = atual
    return dados


def merge_sort(dados, col):
    if len(dados) <= 1:
        return dados.copy()
    meio = len(dados) // 2
    esquerda = merge_sort(dados[:meio], col)
    direita = merge_sort(dados[meio:], col)
    resultado = []
    i = j = 0
    while i < len(esquerda) and j < len(direita):
        if esquerda[i][col] <= direita[j][col]:
            resultado.append(esquerda[i])
            i += 1
        else:
            resultado.append(direita[j])
            j += 1
    resultado.extend(esquerda[i:])
    resultado.extend(direita[j:])
    return resultado


def quick_sort(dados, col):
    if len(dados) <= 1:
        return dados.copy()
    pivo = dados[len(dados) // 2][col]
    menores = [x for x in dados if x[col] < pivo]
    iguais = [x for x in dados if x[col] == pivo]
    maiores = [x for x in dados if x[col] > pivo]
    return quick_sort(menores, col) + iguais + quick_sort(maiores, col)


ALGORITMOS = [
    ("Bubble Sort", bubble_sort),
    ("Selection Sort", selection_sort),
    ("Insertion Sort", insertion_sort),
    ("Merge Sort", merge_sort),
    ("Quick Sort", quick_sort),
]


def esta_ordenado(lista, col):
    return all(lista[k][col] <= lista[k + 1][col] for k in range(len(lista) - 1))


# ==========================================
# 3. EXECUÇÃO DOS EXPERIMENTOS
# ==========================================
planilhas_dados = {}
for rotulo, nome in [("Com filtro", NOME_COM_FILTRO), ("Sem filtro", NOME_SEM_FILTRO)]:
    df, col = carregar_planilha(nome)
    planilhas_dados[rotulo] = (df, col)
    print(f"Planilha '{rotulo}' carregada: {len(df):,} registos.")

tabelas_resultados = {}

for rotulo, (df, col) in planilhas_dados.items():
    print("\n" + "=" * 50)
    print(f"A processar planilha: {rotulo.upper()} ({len(df):,} registos)")
    print("=" * 50)

    dados_inversos = df.sort_values(by=col, ascending=False).to_dict("records")
    dados_aleatorios = df.to_dict("records")
    random.Random(SEMENTE).shuffle(dados_aleatorios)

    resultados = []

    for nome_alg, funcao in ALGORITMOS:
        # Se for o ficheiro sem filtro (500k linhas), ignoramos os O(n²) para o PC não bloquear
        if rotulo == "Sem filtro" and nome_alg in ["Bubble Sort", "Selection Sort", "Insertion Sort"]:
            print(f"A ignorar {nome_alg} no ficheiro Sem filtro (demoraria demasiado)...")
            resultados.append([nome_alg, np.nan, np.nan])
            continue

        print(f"A executar {nome_alg}...")

        t0 = time.perf_counter()
        res_inv = funcao(dados_inversos, col)
        t_inv = time.perf_counter() - t0

        t0 = time.perf_counter()
        res_ale = funcao(dados_aleatorios, col)
        t_ale = time.perf_counter() - t0

        assert esta_ordenado(res_inv, col) and esta_ordenado(res_ale, col)
        resultados.append([nome_alg, t_inv, t_ale])

    tabelas_resultados[rotulo] = pd.DataFrame(resultados, columns=["Algoritmo", "Inversamente ordenada", "Aleatória"])


# ==========================================
# 4. EXIBIÇÃO DAS TABELAS E GRÁFICOS
# ==========================================
for rotulo, tabela in tabelas_resultados.items():
    print(f"\n--- Tabela de Tempos: Planilha {rotulo} ---")
    print(tabela.to_string(index=False))

for rotulo, tabela in tabelas_resultados.items():
    algoritmos = tabela["Algoritmo"].tolist()
    tabela_plot = tabela.fillna(0)
    tempos_inv = tabela_plot["Inversamente ordenada"].tolist()
    tempos_ale = tabela_plot["Aleatória"].tolist()

    x = np.arange(len(algoritmos))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - largura / 2, tempos_inv, largura, label="Inversamente ordenada", color="#C62828")
    ax.bar(x + largura / 2, tempos_ale, largura, label="Aleatória", color="#1565C0")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(algoritmos, fontweight="bold")
    ax.set_title(f"Desempenho dos Algoritmos — Planilha {rotulo}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tempo de execução (segundos) — Log", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    nome_ficheiro = f"grafico_{rotulo.lower().replace(' ', '_')}.png"
    plt.savefig(nome_ficheiro, dpi=150)
    print(f"Gráfico guardado como: {nome_ficheiro}")
    plt.show()

print("\nTudo concluído com sucesso!")
