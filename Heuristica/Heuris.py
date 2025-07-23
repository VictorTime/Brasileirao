import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import matplotlib.pyplot as plt, seaborn as sns
import time
import networkx as nx
from multiprocessing import Pool
import random
import seaborn as sns
from tqdm import tqdm

# Etapa 1: dados dos times
teams = pd.DataFrame([
    {"code": "CAP", "city": "Curitiba, Brasil"},
    {"code": "ATL", "city": "Goiânia, Brasil"},
    {"code": "CAM", "city": "Belo Horizonte, Brasil"},
    {"code": "BAH", "city": "Salvador, Brasil"},
    {"code": "BOT", "city": "Rio de Janeiro, Brasil"},
    {"code": "BRA", "city": "Bragança Paulista, Brasil"},
    {"code": "COR", "city": "São Paulo, Brasil"},
    {"code": "CRI", "city": "Criciúma, Brasil"},
    {"code": "CRU", "city": "Belo Horizonte, Brasil"},
    {"code": "CUI", "city": "Cuiabá, Brasil"},
    {"code": "FLA", "city": "Rio de Janeiro, Brasil"},
    {"code": "FLU", "city": "Rio de Janeiro, Brasil"},
    {"code": "FOR", "city": "Fortaleza, Brasil"},
    {"code": "GRE", "city": "Porto Alegre, Brasil"},
    {"code": "INT", "city": "Porto Alegre, Brasil"},
    {"code": "JUV", "city": "Caxias do Sul, Brasil"},
    {"code": "PAL", "city": "São Paulo, Brasil"},
    {"code": "SAO", "city": "São Paulo, Brasil"},
    {"code": "VAS", "city": "Rio de Janeiro, Brasil"},
    {"code": "VIT", "city": "Salvador, Brasil"},
])

capacity_map = {
    "PAL": 43713, "FLA": 78838, "FLU": 78838, "BOT": 78838, "VAS": 78838,
    "CRU": 66658, "CAM": 66658,
    "FOR": 63903, "BAH": 47907, "VIT": 47907,
    "COR": 47605, "INT": 50128, "GRE": 55225,
    "BRA": 17128, "CUI": 44000,
    "ATL": 14450, "CAM": 66658, "CAP": 42372, 
    "SAO": 72039, "CRI": 20000, "JUV": 20000
}

quali_estadio = {
    "FLA": 0.8,  # Maracanã – histórica, alta capacidade
    "FLU": 0.8,
    "BOT": 0.7,  # Estádio Nilton Santos – moderno
    "VAS": 0.7,
    "PAL": 0.9,  # Allianz Parque – moderno e premiado
    "COR": 0.9,  # Arena Corinthians – moderna, premiada
    "CRU": 0.85, # Mineirão – considerado o melhor da América do Sul
    "CAM": 0.85,
    "GRE": 0.9,  # Arena do Grêmio – única UEFA Category IV no Brasil 
    "INT": 0.8,  # Beira-Rio – moderno
    "FOR": 0.8,  # Castelão – moderno
    "BAH": 0.7,  # Fonte Nova – moderno
    "VIT": 0.7,
    "BRA": 0.6,  # Bragança – pequeno
    "CUI": 0.6,  # Arena Pantanal
    "CAP": 0.7,  # Estádio da Baixada – moderno
    "ATL": 0.6,
    "CRI": 0.5,
    "JUV": 0.5,
    "SAO": 0.85, # Morumbi – grande e tradicional
}

#Baseado na % de visto em https://www.cnnbrasil.com.br/esportes/futebol/pesquisa-revela-maiores-torcidas-do-brasil-veja-ranking/
# Quem eu n achei, vai ter 0,01
relev_map = {
    "FLA": 0.248, "COR": 0.194, "PAL": 0.081, "SAO": 0.077,
    "VAS": 0.048, "GRE": 0.044, "CAM": 0.040, "CRI": 0.031,
    "INT": 0.029, "CRU": 0.027, "BAH": 0.025, "FLU": 0.024,
    "BOT": 0.021, "VIT": 0.019
}


derbies = {
    ("FLA","FLU"),  # Fla-Flu — Rio, maior da América 
    ("FLA","VAS"),  # Clássico dos Milhões — Fla x Vasco
    ("GRE","INT"),  # Grenal 
    ("COR","PAL"),  # Derby Paulista — SP
    ("FLA","COR"),  # Classico das Nações
}

# o valor do classico tava aumentando muito, ai tem os meninos ai pra controlar a influencia deles
alpha, beta = 0.1, 0.05

def compute_relev(code, qual_factor, oponente):
    """
    Verfica a relevância de um jogo baseado na: 
    relevância do time : tamanho da torcida
    derbies: Se for Classico/Rival, 1, se não 0
    quali_estadio: qualidade do estadio * 0,05 (pra não supervalorizar)
    """
    base = relev_map.get(code, 0.01)
    rival = 1 if any((code,o) in derbies or (o,code) in derbies for o in oponente) else 0
    return min(1, base + alpha * rival + beta * qual_factor)

class TimeF():

    def __init__(self, relevancia, nome, cidade):
        self.relavancia = relevancia
        self.nome = nome
        self.cidade = cidade

class Estadio():

    def __init__(self, capacidade, cidade, dias_semana:list ):
        self.capacidade = capacidade
        self.cidade = cidade
        self.dias_semana = dias_semana

        self.em_casa = 0.6
        self.fora_casa = 0.4
    
    def bilheteria(self, time1: TimeF, time2: TimeF, dia:int=0, valor_ingresso_base: float = 52.26):
        t1 = 0
        if time1.cidade == time2.cidade:
            capacidade_time1 = capacidade_time2 = self.capacidade / 2
        else:
            if time1.cidade == self.cidade:
                t1=1
                capacidade_time1 = self.capacidade * self.em_casa
                capacidade_time2 = self.capacidade * self.fora_casa
            else:
                capacidade_time2 = self.capacidade * self.em_casa
                capacidade_time1 = self.capacidade * self.fora_casa

        if(t1==1):
            quali1 = quali_estadio.get(time1.nome, 0)
        else:
            quali1 = quali_estadio.get(time2.nome, 0)
        # 3) chama compute_relev para cada perspectiva
        rel1 = compute_relev(time1.nome, quali1, time2.nome)
        rel2 = compute_relev(time2.nome, quali1, time1.nome)

        # 4) bilheterias ponderadas
        bilheteria1 = capacidade_time1 * rel1
        bilheteria2 = capacidade_time2 * rel2

        # 5) receita total
        fator_dia = self.dias_semana[dia]
        return (bilheteria1 + bilheteria2) * valor_ingresso_base * fator_dia

#Preencher para todos
rel_dict = {t["code"]: relev_map.get(t["code"], 0.001) for _, t in teams.iterrows()}

#criando os times usando o que tem la dataframe
teams_objs = [
    TimeF(
        relevancia=rel_dict[row.code],   # float
        nome=row.code,                   # string
        cidade=row.city                  # string
    )
    for _, row in teams.iterrows()
]

dias_semanas=[0.6, 0.6, 1, 1.1]

# a gente, em tese, tem que fazer o capacidade do estadio variar
estadio = {
    t.nome: Estadio(
        capacity_map.get(t.nome, 20000),  # se não existir no map, usa 20000
        t.cidade,
        dias_semanas
    )
    for t in teams_objs
}


# Etapa 2: geolocalização com fallback
geoloc = Nominatim(user_agent="roundrobin_sched")
coords = {}
for t in teams_objs:
    loc = geoloc.geocode(t.cidade, timeout=10)
    time.sleep(1)
    coords[t.nome] = (loc.latitude, loc.longitude) if loc else (0,0)


# Etapa 4: custo por km e matriz de custos (ANAC)
C_PER_KM = 0.30

n = len(teams_objs)
cost_mat = np.zeros((n,n))
for i,t1 in enumerate(teams_objs):
    for j,t2 in enumerate(teams_objs):
        if i == j: continue
        d = geodesic(coords[t1.nome], coords[t2.nome]).km
        cost_mat[i,j] = 2 * d * C_PER_KM

# Etapa 5: algoritmo Round-Robin espelhado
def round_robin_pairing(team_ids):
    ids = list(random.sample(team_ids,len(team_ids)))
    if len(ids) % 2 == 1: ids.append(None)
    n = len(ids)
    rounds = []
    for r in range(n-1):
        l1 = ids[:n//2]
        l2 = ids[n//2:][::-1]
        rounds.append(list(zip(l1, l2)))
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return rounds

base_rounds = round_robin_pairing(list(range(n)))
mirrored = [[(b, a) for (a, b) in rnd] for rnd in base_rounds]
schedule = base_rounds + mirrored  # 38 rodadas

def lucro_schedule(schedule):
    total = 0
    for r_idx, rodada in enumerate(schedule):
        dia = r_idx % len(dias_semanas)
        for hid, aid in rodada:
            t_casa = teams_objs[hid]
            t_fora = teams_objs[aid]
            est = estadio[t_casa.nome]
            rev = est.bilheteria(t_casa, t_fora, dia)
            cost = cost_mat[hid, aid]
            total += rev - cost
    return total

def ajusta_homeaway(schedule, max_away=3):
    """
    Garante que nenhum time terá mais de `max_away` jogos fora seguidos,
    invertendo mandante/visitante se necessário.
    """
    home_away = {i: [] for i in range(n)}

    for rnd in schedule:
        for a, b in rnd:
            if None in (a, b): continue
            home_away[a].append('home')
            home_away[b].append('away')

    # Corrige sequências maiores que max_away
    for team_id, seq in home_away.items():
        count = 0
        for i in range(len(seq)):
            if seq[i] == 'away':
                count += 1
                if count > max_away:
                    # Tenta inverter o jogo nessa rodada
                    for rnd in schedule[i:]:
                        for idx, (a, b) in enumerate(rnd):
                            if a == team_id:
                                rnd[idx] = (b, a)
                                break
                            elif b == team_id:
                                rnd[idx] = (a, b)
                                break
                    break
            else:
                count = 0
    return schedule

schedule = ajusta_homeaway(schedule)

# ——— Heurísticas de otimização ———————————————————

def swap_and_eval_cached(schedule, iters=200):
    """
    Heurística de swap com cache de lucros, recalculando internamente os lucros por rodada.
    Retorna: best_schedule, best_profit, best_round_profit_list
    """
    # round_profit inicial calculado a partir do schedule recebido
    round_profit = [lucro_schedule([rnd]) for rnd in schedule]
    best = [r[:] for r in schedule]
    best_profit = sum(round_profit)
    best_round_profit = round_profit.copy()

    for _ in range(iters):
        i, j = np.random.choice(len(best), 2, replace=False)
        new_sched = [r[:] for r in best]
        new_sched[i], new_sched[j] = new_sched[j], new_sched[i]

        # Recalcula apenas os lucros das duas rodadas trocadas
        new_ri = lucro_schedule([new_sched[i]])
        new_rj = lucro_schedule([new_sched[j]])

        candidate_profit = best_profit - best_round_profit[i] - best_round_profit[j] + new_ri + new_rj

        if candidate_profit > best_profit:
            best = new_sched
            best_profit = candidate_profit
            best_round_profit[i], best_round_profit[j] = new_ri, new_rj

    return best, best_profit, best_round_profit

def heuristic_greedy_swap(schedule, iters=200):
    best, best_profit = schedule, lucro_schedule(schedule)
    for _ in range(iters):
        new = [r[:] for r in best]
        i, j = np.random.choice(len(new), 2, replace=False)
        # Troca times 
        new[i], new[j] = new[j], new[i]
        p = lucro_schedule(new)
        if p > best_profit:
            best, best_profit = new, p
    return best, best_profit

def heuristic_flip_homeaway(schedule, iters=200):
    best, best_profit = schedule, lucro_schedule(schedule)
    for _ in range(iters):
        new = [r[:] for r in best]
        rnd = np.random.randint(len(new))
        idx = np.random.randint(len(new[rnd]))
        a, b = new[rnd][idx]
        new[rnd][idx] = (b, a)
        p = lucro_schedule(new)
        if p > best_profit:
            best, best_profit = new, p
    return best, best_profit

def flip_and_eval_cached(schedule, iters=200):
    """
    Heurística de flip com cache de lucros, recalculando internamente os lucros por rodada.
    Retorna: best_schedule, best_profit, best_round_profit_list
    """
    round_profit = [lucro_schedule([rnd]) for rnd in schedule]
    best = [r[:] for r in schedule]
    best_round_profit = round_profit.copy()
    best_profit = sum(best_round_profit)

    for _ in range(iters):
        rnd_idx = np.random.randint(len(best))
        pos = np.random.randint(len(best[rnd_idx]))
        new_sched = [r[:] for r in best]
        a, b = new_sched[rnd_idx][pos]
        new_sched[rnd_idx][pos] = (b, a)

        new_r = lucro_schedule([new_sched[rnd_idx]])
        candidate = best_profit - best_round_profit[rnd_idx] + new_r

        if candidate > best_profit:
            best = new_sched
            best_profit = candidate
            best_round_profit[rnd_idx] = new_r

    return best, best_profit, best_round_profit

def heuristic_restrict_shuffle(schedule, window=4, iters=100):
    """
    Heurística que embaralha o conjunto de 'window' rodadas consecutivas.
    Retorna: best_schedule, best_profit
    """
    best = [r[:] for r in schedule]
    best_profit = lucro_schedule(best)
    L = len(best)

    for _ in range(iters):
        new_sched = [r[:] for r in best]
        start = np.random.randint(0, L - window + 1)
        sub = new_sched[start:start+window]
        random.shuffle(sub)
        new_sched[start:start+window] = sub

        p = lucro_schedule(new_sched)
        if p > best_profit:
            best, best_profit = new_sched, p

    return best, best_profit

def small_greedy_move(cur, round_profit):
    """
    Gera os 3 movimentos (swap, flip, shuffle),
    calcula o lucro e retorna o que gerar o maior ganho (delta).
    """
    best_new = None
    best_new_rp = None
    best_delta = float('-inf')
    base_profit = sum(round_profit)

    # 1) Swap de rodadas
    new = [rnd[:] for rnd in cur]
    i, j = np.random.choice(len(new), 2, replace=False)
    new[i], new[j] = new[j], new[i]
    rp = [lucro_schedule([rnd]) for rnd in new]
    delta = sum(rp) - base_profit
    if delta > best_delta:
        best_delta, best_new, best_new_rp = delta, new, rp

    # 2) Flip de mandante/visitante
    new = [rnd[:] for rnd in cur]
    rnd_idx = np.random.randint(len(new))
    pos = np.random.randint(len(new[rnd_idx]))
    a, b = new[rnd_idx][pos]
    new[rnd_idx][pos] = (b, a)
    rp = [lucro_schedule([rnd]) for rnd in new]
    delta = sum(rp) - base_profit
    if delta > best_delta:
        best_delta, best_new, best_new_rp = delta, new, rp

    # 3) Shuffle local de bloco
    new = [rnd[:] for rnd in cur]
    w = 3
    start = np.random.randint(0, len(new) - w)
    sub = new[start:start+w]
    np.random.shuffle(sub)
    new[start:start+w] = sub
    rp = [lucro_schedule([rnd]) for rnd in new]
    delta = sum(rp) - base_profit
    if delta > best_delta:
        best_delta, best_new, best_new_rp = delta, new, rp

    return best_new, best_new_rp, best_delta

# —– Lucro delta incremental —–❐
def lucro_delta(cur, new, round_profit, idxs):
    i, j = idxs
    new_ri = lucro_schedule([new[i]])
    new_rj = lucro_schedule([new[j]])
    delta = new_ri + new_rj - round_profit[i] - round_profit[j]
    return delta, new_ri, new_rj

# —– Atualização do round_profit —–❐
def update_round_profit(round_profit, new_vals, idxs):
    i, j = idxs
    new_ri, new_rj = new_vals
    round_profit[i], round_profit[j] = new_ri, new_rj

# —– Simulated Annealing —–❐

def simulate_annealing_full(schedule, T0=5000, alpha=0.999, iters=5000):

    # Estado atual
    cur = [r[:] for r in schedule]
    cur_rp = [lucro_schedule([rnd]) for rnd in cur]
    cur_profit = sum(cur_rp)

    # Melhor estado
    best = [r[:] for r in cur]
    best_rp = cur_rp.copy()
    best_profit = cur_profit

    T = T0
    T_min = 1e-6

    for _ in range(iters):
        new_sched, new_rp, delta= small_greedy_move(cur,cur_rp)
        new_profit = sum(new_rp)
        # Critério de aceitação
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            cur, cur_rp, cur_profit = new_sched, new_rp, new_profit
            # Atualiza melhor se necessário
            if cur_profit > best_profit:
                best, best_rp, best_profit = cur, cur_rp.copy(), cur_profit
        # Esfriamento
        T = max(T_min, T * alpha)

    return best, best_profit, best_rp


def evaluate_heuristics(base_schedule, base_round_profit, heuristics: list, names: list):
    """
    Executa cada heurística isoladamente (partindo do mesmo calendário inicial),
    mede tempo e lucro final, e também aplica todas em sequência para comparar.
    Retorna um DataFrame com as colunas: ['heuristica','lucro','tempo_s'].
    """
    results = []

    # 1) avaliações individuais
    for fn, name in zip(heuristics, names):
        sched_copy = [r[:] for r in base_schedule]
        start = time.perf_counter()
        new_sched, profit = fn(sched_copy)
        elapsed = time.perf_counter() - start
        results.append({'heuristica': name, 'lucro': profit, 'tempo_s': elapsed})

    # 2) avaliação da heurística combinada (aplica todas em sequência)
    sched = [r[:] for r in base_schedule]
    total_start = time.perf_counter()
    for fn in heuristics:
        sched, _ = fn(sched)  # cada heurística retorna (sched, lucro)
    combined_profit = lucro_schedule(sched)
    combined_time = time.perf_counter() - total_start
    results.append({'heuristica': 'combinada', 'lucro': combined_profit, 'tempo_s': combined_time})

    return pd.DataFrame(results)


# ------- EVALUATE ---------
def calcular_lucros_confrontos(schedule, teams_objs):
    dados = []
    for r_idx, rodada in enumerate(schedule):
        for hid, aid in rodada:
            if None in (hid, aid): continue
            t_casa = teams_objs[hid]
            t_fora = teams_objs[aid]
            est = estadio[t_casa.nome]
            rev = est.bilheteria(t_casa, t_fora, r_idx % len(dias_semanas))
            cost = cost_mat[hid, aid]
            lucro = rev - cost
            dados.append({
                "rodada": r_idx + 1,
                "mandante": t_casa.nome,
                "visitante": t_fora.nome,
                "lucro": lucro
            })
    return pd.DataFrame(dados)

def plot_metrics_for_schedule(schedule, name):
    """
    Gera e mostra:
     - Top/Bottom confrontos
     - Top/Bottom rodadas
     - Receita total por time
    para um dado calendário `schedule` e identifica pelo `name`.
    """
    # recalcula todos os confrontos
    df = calcular_lucros_confrontos(schedule, teams_objs)

    # Top/Bottom Confrontos
    best_c = df.nlargest(10, 'lucro')
    worst_c = df.nsmallest(10, 'lucro')

    # Lucro por rodada
    lucros_r = df.groupby('rodada')['lucro'].sum().reset_index()
    best_r = lucros_r.nlargest(5, 'lucro')
    worst_r = lucros_r.nsmallest(5, 'lucro')

    # Receita por time
    df_home = df.groupby('mandante')['lucro'].sum().reset_index().rename(
        columns={'mandante':'time','lucro':'home_lucro'})
    df_away = df.groupby('visitante')['lucro'].sum().reset_index().rename(
        columns={'visitante':'time','lucro':'away_lucro'})
    df_time = pd.merge(df_home, df_away, on='time', how='outer').fillna(0)
    df_time['total_lucro'] = df_time['home_lucro'] + df_time['away_lucro']

    sns.set(style='whitegrid')

    # --- Plots ---
    # 1) Top 10 Confrontos
    plt.figure(figsize=(10,6))
    sns.barplot(
        x='lucro', y='mandante', hue='visitante',
        data=best_c.sort_values('lucro', ascending=True)
    )
    plt.title(f'Top 10 Confrontos Mais Lucrativos ({name})')
    plt.xlabel('Lucro (R$)')
    plt.ylabel('Mandante')
    plt.legend(title='Visitante', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # 2) Bottom 10 Confrontos
    plt.figure(figsize=(10,6))
    sns.barplot(
        x='lucro', y='mandante', hue='visitante',
        data=worst_c.sort_values('lucro', ascending=False)
    )
    plt.title(f'Top 10 Confrontos Menos Lucrativos ({name})')
    plt.show()

    # 3) Top 5 Rodadas
    plt.figure(figsize=(8,4))
    sns.barplot(x='rodada', y='lucro', data=best_r)
    plt.title(f'Top 5 Rodadas Mais Lucrativas ({name})')
    plt.show()

    # 4) Bottom 5 Rodadas
    plt.figure(figsize=(8,4))
    sns.barplot(x='rodada', y='lucro', data=worst_r)
    plt.title(f'Top 5 Rodadas Menos Lucrativas ({name})')
    plt.show()

    # 5) Receita Total por Time
    plt.figure(figsize=(10,8))
    sns.barplot(
        x='total_lucro', y='time',
        data=df_time.sort_values('total_lucro', ascending=False)
    )
    plt.title(f'Receita Total por Time ({name})')
    plt.xlabel('Lucro Total (R$)')
    plt.ylabel('Time')
    plt.tight_layout()
    plt.show()

def run_tests(fn, base_schedule, round_profit, repeats):
    """
    Executa `fn(schedule)` `repeats` vezes, retornando as estatísticas
    de lucro (mean, std, min, max).
    `fn` deve receber apenas `schedule` e devolver (best_schedule, best_profit).
    """
    profits = []
    for _ in range(repeats):
        sched_copy = [r[:] for r in base_schedule]
        # para as heurísticas que precisem de round_profit, caminhamos com o global
        _, p = fn(sched_copy)
        profits.append(p)
    arr = np.array(profits)
    return {
        'repeats': repeats,
        'mean': float(arr.mean()),
        'std':   float(arr.std()),
        'min':   float(arr.min()),
        'max':   float(arr.max())
    }

# --- Combinações específicas de heurísticas ---
def combo_greedy_flip(schedule):
    sched = [r[:] for r in schedule]
    sched, _ = heuristic_greedy_swap(sched, iters=200)
    sched, _ = heuristic_flip_homeaway(sched, iters=200)
    return sched, lucro_schedule(sched)

def combo_greedy_flip_shuffle(schedule):
    sched = [r[:] for r in schedule]
    sched, _ = heuristic_greedy_swap(sched, iters=200)
    sched, _ = heuristic_flip_homeaway(sched, iters=200)
    sched, _ = heuristic_restrict_shuffle(sched, window=4, iters=100)
    return sched, lucro_schedule(sched)

def combo_flip_shuffle(schedule):
    sched = [r[:] for r in schedule]
    sched, _ = heuristic_flip_homeaway(sched, iters=200)
    sched, _ = heuristic_restrict_shuffle(sched, window=4, iters=100)
    return sched, lucro_schedule(sched)

def combo_greedy_shuffle(schedule):
    sched = [r[:] for r in schedule]
    sched, _ = heuristic_greedy_swap(sched, iters=200)
    sched, _ = heuristic_restrict_shuffle(sched, window=4, iters=100)
    return sched, lucro_schedule(sched)

def combo_shuffle_sim_anneal(schedule):
    sched = [r[:] for r in schedule]
    sched, _ = heuristic_restrict_shuffle(sched, window=4, iters=100)
    sched, _ = simulate_annealing_full(sched, T0=1000, alpha=0.999, iters=5000)[:2]
    return sched, lucro_schedule(sched)

def combo_greedy_sim_anneal(schedule):
    sched = [r[:] for r in schedule]
    sched, _ = heuristic_greedy_swap(sched, iters=200)
    sched, _ = simulate_annealing_full(sched, T0=1000, alpha=0.999, iters=5000)[:2]
    return sched, lucro_schedule(sched)

def combo_all(schedule):
    sched = [r[:] for r in schedule]
    # Aplica todas as seis heurísticas em sequência
    sched, _ = heuristic_greedy_swap(sched, iters=200)
    sched, _ = swap_and_eval_cached(sched, iters=200)[:2]
    sched, _ = heuristic_flip_homeaway(sched, iters=200)
    sched, _ = flip_and_eval_cached(sched, iters=200)[:2]
    sched, _ = heuristic_restrict_shuffle(sched, window=4, iters=100)
    sched, _ = simulate_annealing_full(sched, T0=1000, alpha=0.999, iters=5000)[:2]
    return sched, lucro_schedule(sched)

# --- Atualização das listas de testes ---
heuristic_fns = [
    lambda sch: heuristic_greedy_swap(sch, iters=200),
    lambda sch: swap_and_eval_cached(sch, iters=200)[:2],
    lambda sch: heuristic_flip_homeaway(sch, iters=200),
    lambda sch: flip_and_eval_cached(sch, iters=200)[:2],
    lambda sch: heuristic_restrict_shuffle(sch, window=4, iters=100),
    lambda sch: simulate_annealing_full(sch, T0=1000, alpha=0.999, iters=5000)[:2],
    combo_greedy_flip,
    combo_greedy_flip_shuffle,
    combo_flip_shuffle,
    combo_greedy_shuffle,
    combo_shuffle_sim_anneal,
    combo_greedy_sim_anneal,
    combo_all
]
heuristic_names = [
    "greedy_swap", "swap_cache", "flip", "flip_cache", "shuffle", "sim_anneal",
    "greedy_flip", "greedy_flip_shuffle", "flip_shuffle", "greedy_shuffle", "shuffle_sim_anneal", "greedy_sim_anneal",
    "combined"
]


base_schedule = schedule
round_profit = [lucro_schedule([rnd]) for rnd in base_schedule]  # lucro por rodada
current_profit = sum(round_profit)


results = []
for fn, name in tqdm(zip(heuristic_fns, heuristic_names), 
                     desc="Heurísticas", total=len(heuristic_fns)):
    for rep in tqdm([1, 5, 10, 50], desc=f"{name}", leave=False):
        stats = run_tests(fn=fn, base_schedule=base_schedule, round_profit=round_profit, repeats=rep)
        stats['heuristica'] = name
        stats['repeats']     = rep
        results.append(stats)

# --- 1) Monta o DataFrame de estatísticas ---
df_stats = pd.DataFrame(results)

# Pivot para ter mean_X e std_X para cada repeats X
pivot = df_stats.pivot(index='heuristica', columns='repeats')[['mean','std']]
pivot.columns = [f"{stat}_{rep}" for stat, rep in pivot.columns]
tabela_final = pivot.reset_index()

print(tabela_final)


# --- 2) Gera os gráficos individuais de cada heurística ---
for fn, name in zip(heuristic_fns, heuristic_names):
    sched, _ = fn([r[:] for r in base_schedule])         # aplica a heurística partindo do base_schedule
    plot_metrics_for_schedule(sched, name)               # plota Top/Bottom Confrontos, Rodadas e Receita por Time


# --- 3) Chama evaluate_heuristics para todas as heurísticas ---
# (recria round_profit internamente)
df_bench = evaluate_heuristics(
    base_schedule,
    # recalcula round_profit inicial
    [lucro_schedule([rnd]) for rnd in base_schedule],
    heuristic_fns,
    heuristic_names
)
print(df_bench)

# names = [n for _, n in heuristic_fns]
# fns   = [f for f, _ in heuristic_fns]
# df_stats = pd.DataFrame(results)

# # Pivot para mostrar mean_X e std_X
# pivot = df_stats.pivot(index='heuristica', columns='repeats')[['mean', 'std']]
# pivot.columns = [f"{stat}_{rep}" for stat, rep in pivot.columns]
# tabela_final = pivot.reset_index()

# print(tabela_final)

# df_stats = pd.DataFrame(results)

# # Pivot para ter colunas mean_X e std_X para cada número de repetições X
# pivot = df_stats.pivot(index='heuristica', columns='repeats')[['mean', 'std']]

# # Opcional: renomear as colunas para algo como mean_1, std_1, mean_5, std_5…
# pivot.columns = [f"{stat}_{rep}" for stat, rep in pivot.columns]

# # Reset no índice para exibir como tabela “plana”
# tabela_final = pivot.reset_index()

# print(tabela_final)
# df_stats = pd.DataFrame(results)

# sched_comb = base_schedule.copy()
# for fn in fns:
#     sched_comb, _ = fn(sched_comb)
# plot_metrics_for_schedule(sched_comb, 'combinada')


# df_bench = evaluate_heuristics(base_schedule, round_profit, fns, names)
# print(df_bench)



























# history = []
# current = schedule
# current_profit = lucro_schedule(current)
# history.append(("Inicial", current_profit))

# # Swap simples O(n²)
# current, p1 = heuristic_greedy_swap(current, iters=1000)
# round_profit = [lucro_schedule([rnd]) for rnd in current]
# history.append(("Swap", p1))

# # Swap com cache O(n)
# current, p2, round_profit = swap_and_eval_cached(current, round_profit, iters=1000)
# history.append(("Swap_Cache", p2))

# # Flip simples O(n²)
# current, p3 = heuristic_flip_homeaway(current, iters=1000)
# round_profit = [lucro_schedule([rnd]) for rnd in current]
# history.append(("Flip", p3))

# # Flip com cache O(1)
# current, p4, round_profit = flip_and_eval_cached(current, round_profit, iters=1000)
# history.append(("Flip_Cache", p4))

# # SA
# round_profit = [lucro_schedule([rnd]) for rnd in current]
# current, p6, final_rp = simulate_annealing_full(current, round_profit)
# history.append(("sa", p6))

# # Shuffle (use lucro calculado diretamente)
# current, p5 = heuristic_restrict_shuffle(current, window=6, iters=1000)
# history.append(("Shuffle", p5))

# def calcular_lucros_confrontos(schedule, teams_objs):
#     dados = []
#     for r_idx, rodada in enumerate(schedule):
#         for hid, aid in rodada:
#             if None in (hid, aid): continue
#             t_casa = teams_objs[hid]
#             t_fora = teams_objs[aid]
#             est = estadio[t_casa.nome]
#             rev = est.bilheteria(t_casa, t_fora, r_idx % len(dias_semanas))
#             cost = cost_mat[hid, aid]
#             lucro = rev - cost
#             dados.append({
#                 "rodada": r_idx + 1,
#                 "mandante": t_casa.nome,
#                 "visitante": t_fora.nome,
#                 "lucro": lucro
#             })
#     return pd.DataFrame(dados)

# # --- 1. Lucros por confronto e rodada ---
# df = calcular_lucros_confrontos(current, teams_objs)
# df['lucro'] = df['lucro'].astype(float)

# melhores_confrontos = df.nlargest(10, 'lucro')
# piores_confrontos = df.nsmallest(10, 'lucro')

# lucros_por_rodada = df.groupby('rodada')['lucro'].sum().reset_index()
# melhores_rodadas = lucros_por_rodada.nlargest(5, 'lucro')
# piores_rodadas    = lucros_por_rodada.nsmallest(5, 'lucro')

# calendario_final = df[['rodada','mandante','visitante']].drop_duplicates().sort_values('rodada')

# sns.set(style='whitegrid', palette='muted')

# # a) Lucro por rodada
# plt.figure(figsize=(12,4))
# sns.barplot(x='rodada', y='lucro', data=lucros_por_rodada)
# plt.title('Lucro por Rodada')
# plt.xlabel('Rodada')
# plt.ylabel('Receita Líquida (R$)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # b) Top 5 melhores e piores rodadas
# plt.figure(figsize=(10,4))
# sns.barplot(x='rodada', y='lucro', data=melhores_rodadas, color='green')
# plt.title('Top 5 Rodadas Mais Lucrativas')
# plt.show()

# plt.figure(figsize=(10,4))
# sns.barplot(x='rodada', y='lucro', data=piores_rodadas, color='red')
# plt.title('Top 5 Rodadas Menos Lucrativas')
# plt.show()

# # c) Top Confrontos
# plt.figure(figsize=(10,6))
# sns.barplot(x='lucro', y='mandante', hue='visitante',
#             data=melhores_confrontos.sort_values('lucro', ascending=True))
# plt.title('Top 10 Confrontos Mais Lucrativos')
# plt.xlabel('Lucro (R$)')
# plt.ylabel('Mandante / Visitante')
# plt.legend(title='Visitante')
# plt.tight_layout()
# plt.show()

# # calcular receita acumulada por time em casa e fora
# df_home = df.groupby('mandante')['lucro'].sum().reset_index().rename(columns={'mandante':'time','lucro':'home_lucro'})
# df_away = df.groupby('visitante')['lucro'].sum().reset_index().rename(columns={'visitante':'time','lucro':'away_lucro'})
# df_time = pd.merge(df_home, df_away, on='time', how='outer').fillna(0)
# df_time['total_lucro'] = df_time['home_lucro'] + df_time['away_lucro']

# # d) gráfico receita por time
# plt.figure(figsize=(12,6))
# sns.barplot(x='total_lucro', y='time', data=df_time.sort_values('total_lucro', ascending=False))
# plt.title('Receita Total por Time')
# plt.xlabel('Lucro Total (R$)')
# plt.ylabel('Time')
# plt.tight_layout()
# plt.show()

# print("\nMelhores Confrontos\n", melhores_confrontos[['mandante','visitante','lucro']])
# print("\nPiores Confrontos\n", piores_confrontos[['mandante','visitante','lucro']])
# print("\nMelhores Rodadas\n", melhores_rodadas)
# print("\nPiores Rodadas\n", piores_rodadas)
# print("\nReceita por Time\n", df_time.sort_values('total_lucro', ascending=False).head(10))
# print("\nCalendário (primeiras 20 partidas)\n", calendario_final.head(20))

# # --- Histórico e Visualização Final ---
# history = [
#     ("Inicial", current_profit),
#     ("Swap", p1),
#     ("Swap_Cache", p2),
#     ("Flip", p3),
#     ("Flip_Cache", p4),
#     ("SimulatedAnnealing", p6),
#     ("Shuffle", p5),
# ]

# steps, profits = zip(*history)

# plt.figure(figsize=(10,6))
# plt.plot(steps, profits, marker='o', linestyle='-', color='purple')
# plt.title('Evolução do Lucro Estimado por Etapa')
# plt.xlabel('Etapa')
# plt.ylabel('Lucro Estimado (R$)')
# plt.grid(True)
# for i, (step_name, profit) in enumerate(history):
#     plt.annotate(f"R$ {profit:,.0f}", (i, profit),
#                  textcoords="offset points", xytext=(0,8),
#                  ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
# plt.tight_layout()
# plt.show()

# # --- Logs Claros dos Resultados ---
# print("🔍 Resultados por etapa:")
# print(f"Inicial            : R$ {history[0][1]:,.2f}")
# print(f"Swap (sem cache)   : R$ {history[1][1]:,.2f}")
# print(f"Swap com Cache     : R$ {history[2][1]:,.2f}")
# print(f"Flip (sem cache)   : R$ {history[3][1]:,.2f}")
# print(f"Flip com Cache     : R$ {history[4][1]:,.2f}")
# print(f"Simulated Annealing: R$ {history[6][1]:,.2f}")
# print(f"Shuffle final      : R$ {history[5][1]:,.2f}")
