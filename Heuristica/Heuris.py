import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import matplotlib.pyplot as plt, seaborn as sns

import networkx as nx
from multiprocessing import Pool
import random

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

        if time1.cidade == time2.cidade:
            capacidade_time1 = capacidade_time2 = self.capacidade / 2
        else:
            if time1.cidade == time2.cidade:
                capacidade_time1 = self.capacidade * self.em_casa
                capacidade_time2 = self.capacidade * self.fora_casa
            else:
                capacidade_time2 = self.capacidade * self.em_casa
                capacidade_time1 = self.capacidade * self.fora_casa

        bilheteria_time1 = capacidade_time1 * time1.relavancia
        bilheteria_time2 = capacidade_time2 * time2.relavancia

        bilheteria_total = (bilheteria_time1 + bilheteria_time2) * valor_ingresso_base * self.dias_semana[dia]

        return bilheteria_total

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
        ids = list(random.sample(ids,len(ids)))
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

def swap_and_eval_cached(schedule, round_profit, iters=200):
    best = schedule
    best_profit = sum(round_profit)
    best_round_profit = round_profit.copy()

    for _ in range(iters):
        i, j = np.random.choice(len(schedule), 2, replace=False)
        new = [r[:] for r in best]
        new[i], new[j] = new[j], new[i]

        # Recalcula apenas os lucros das duas rodadas trocadas
        new_ri = lucro_schedule([new[i]])
        new_rj = lucro_schedule([new[j]])

        candidate_profit = best_profit - best_round_profit[i] - best_round_profit[j] + new_ri + new_rj

        if candidate_profit > best_profit:
            best = new
            best_profit = candidate_profit
            best_round_profit[i], best_round_profit[j] = new_ri, new_rj

    return best, best_profit, best_round_profit

# ——— Heurísticas de otimização ———————————————————

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

def flip_and_eval_cached(schedule, round_profit, iters=200):
    best = schedule
    best_round_profit = round_profit.copy()
    best_profit = sum(best_round_profit)
    for _ in range(iters):
        rnd_idx = np.random.randint(len(schedule))
        pos = np.random.randint(len(schedule[rnd_idx]))
        new_sched = [r[:] for r in best]
        a, b = new_sched[rnd_idx][pos]
        new_sched[rnd_idx][pos] = (b, a)

        new_r = lucro_schedule([new_sched[rnd_idx]])
        cand = best_profit - best_round_profit[rnd_idx] + new_r

        if cand > best_profit:
            best = new_sched
            best_profit = cand
            best_round_profit[rnd_idx] = new_r

    return best, best_profit, best_round_profit

def heuristic_restrict_shuffle(schedule, window=4, iters=100):
    best, best_profit = schedule, lucro_schedule(schedule)
    L = len(best)
    for _ in range(iters):
        new = [r[:] for r in best]
        start = np.random.randint(0, L - window)
        sub = new[start:start+window]
        np.random.shuffle(sub)
        new[start:start+window] = sub
        p = lucro_schedule(new)
        if p > best_profit:
            best, best_profit = new, p
    return best, best_profit

def small_random_move(cur):
    new = [r[:] for r in cur]
    op = np.random.choice(['swap', 'flip', 'shuffle'])
    if op == 'swap':
        i, j = np.random.choice(len(new), 2, replace=False)
        new[i], new[j] = new[j], new[i]
    elif op == 'flip':
        rnd = np.random.randint(len(new))
        pos = np.random.randint(len(new[rnd]))
        a,b = new[rnd][pos]; new[rnd][pos] = (b,a)
    else:
        w = 3
        start = np.random.randint(0, len(new)-w)
        sub = new[start:start+w]; np.random.shuffle(sub)
        new[start:start+w] = sub
    return new

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

def simulate_annealing_full(schedule, round_profit, T0=1000, alpha=0.995, iters=5000):
    cur, rp = schedule, round_profit
    cur_profit = sum(rp)
    best, best_profit = cur, cur_profit
    T = T0
    T_min = 1e-6

    for _ in range(iters):
        new = small_random_move(cur)
        # detectar modificaçōes:
        # para simplicidade, recalcula todos lucros de rodada
        new_round_profit = [lucro_schedule([rnd]) for rnd in new]
        delta = sum(new_round_profit) - cur_profit

        if delta > 0 or np.random.rand() < np.exp(delta / T):
            cur, rp, cur_profit = new, new_round_profit, cur_profit + delta
            if cur_profit > best_profit:
                best, best_profit = cur, cur_profit
        T = max(T_min, T * alpha)

    return best, best_profit, rp

base_schedule = schedule
round_profit = [lucro_schedule([rnd]) for rnd in base_schedule]  # lucro por rodada
current_profit = sum(round_profit)

history = []
current = schedule
current_profit = lucro_schedule(current)
history.append(("Inicial", current_profit))

# Swap simples O(n²)
current, p1 = heuristic_greedy_swap(current, iters=1000)
round_profit = [lucro_schedule([rnd]) for rnd in current]
history.append(("Swap", p1))

# Swap com cache O(n)
current, p2, round_profit = swap_and_eval_cached(current, round_profit, iters=1000)
history.append(("Swap_Cache", p2))

# Flip simples O(n²)
current, p3 = heuristic_flip_homeaway(current, iters=1000)
round_profit = [lucro_schedule([rnd]) for rnd in current]
history.append(("Flip", p3))

# Flip com cache O(1)
current, p4, round_profit = flip_and_eval_cached(current, round_profit, iters=1000)
history.append(("Flip_Cache", p4))

# Shuffle (use lucro calculado diretamente)
current, p5 = heuristic_restrict_shuffle(current, window=4, iters=1000)
history.append(("Shuffle", p5))

# SA
round_profit = [lucro_schedule([rnd]) for rnd in current]
current, p6, final_rp = simulate_annealing_full(schedule, round_profit)
history.append(("sa", p6))

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

# --- 1. Lucros por confronto e rodada ---
df = calcular_lucros_confrontos(current, teams_objs)
df['lucro'] = df['lucro'].astype(float)

melhores_confrontos = df.nlargest(10, 'lucro')
piores_confrontos = df.nsmallest(10, 'lucro')

lucros_por_rodada = df.groupby('rodada')['lucro'].sum().reset_index()
melhores_rodadas = lucros_por_rodada.nlargest(5, 'lucro')
piores_rodadas    = lucros_por_rodada.nsmallest(5, 'lucro')

calendario_final = df[['rodada','mandante','visitante']].drop_duplicates().sort_values('rodada')

sns.set(style='whitegrid', palette='muted')

# a) Lucro por rodada
plt.figure(figsize=(12,4))
sns.barplot(x='rodada', y='lucro', data=lucros_por_rodada)
plt.title('Lucro por Rodada')
plt.xlabel('Rodada')
plt.ylabel('Receita Líquida (R$)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# b) Top 5 melhores e piores rodadas
plt.figure(figsize=(10,4))
sns.barplot(x='rodada', y='lucro', data=melhores_rodadas, color='green')
plt.title('Top 5 Rodadas Mais Lucrativas')
plt.show()

plt.figure(figsize=(10,4))
sns.barplot(x='rodada', y='lucro', data=piores_rodadas, color='red')
plt.title('Top 5 Rodadas Menos Lucrativas')
plt.show()

# c) Top Confrontos
plt.figure(figsize=(10,6))
sns.barplot(x='lucro', y='mandante', hue='visitante',
            data=melhores_confrontos.sort_values('lucro', ascending=True))
plt.title('Top 10 Confrontos Mais Lucrativos')
plt.xlabel('Lucro (R$)')
plt.ylabel('Mandante / Visitante')
plt.legend(title='Visitante')
plt.tight_layout()
plt.show()

# calcular receita acumulada por time em casa e fora
df_home = df.groupby('mandante')['lucro'].sum().reset_index().rename(columns={'mandante':'time','lucro':'home_lucro'})
df_away = df.groupby('visitante')['lucro'].sum().reset_index().rename(columns={'visitante':'time','lucro':'away_lucro'})
df_time = pd.merge(df_home, df_away, on='time', how='outer').fillna(0)
df_time['total_lucro'] = df_time['home_lucro'] + df_time['away_lucro']

# d) gráfico receita por time
plt.figure(figsize=(12,6))
sns.barplot(x='total_lucro', y='time', data=df_time.sort_values('total_lucro', ascending=False))
plt.title('Receita Total por Time')
plt.xlabel('Lucro Total (R$)')
plt.ylabel('Time')
plt.tight_layout()
plt.show()

print("\nMelhores Confrontos\n", melhores_confrontos[['mandante','visitante','lucro']])
print("\nPiores Confrontos\n", piores_confrontos[['mandante','visitante','lucro']])
print("\nMelhores Rodadas\n", melhores_rodadas)
print("\nPiores Rodadas\n", piores_rodadas)
print("\nReceita por Time\n", df_time.sort_values('total_lucro', ascending=False).head(10))
print("\nCalendário (primeiras 20 partidas)\n", calendario_final.head(20))

# --- Histórico e Visualização Final ---
history = [
    ("Inicial", current_profit),
    ("Swap", p1),
    ("Swap_Cache", p2),
    ("Flip", p3),
    ("Flip_Cache", p4),
    ("Shuffle", p5),
    ("SimulatedAnnealing", p6),
]

steps, profits = zip(*history)

plt.figure(figsize=(10,6))
plt.plot(steps, profits, marker='o', linestyle='-', color='purple')
plt.title('Evolução do Lucro Estimado por Etapa')
plt.xlabel('Etapa')
plt.ylabel('Lucro Estimado (R$)')
plt.grid(True)
for i, (step_name, profit) in enumerate(history):
    plt.annotate(f"R$ {profit:,.0f}", (i, profit),
                 textcoords="offset points", xytext=(0,8),
                 ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
plt.tight_layout()
plt.show()

# --- Logs Claros dos Resultados ---
print("🔍 Resultados por etapa:")
print(f"Inicial            : R$ {history[0][1]:,.2f}")
print(f"Swap (sem cache)   : R$ {history[1][1]:,.2f}")
print(f"Swap com Cache     : R$ {history[2][1]:,.2f}")
print(f"Flip (sem cache)   : R$ {history[3][1]:,.2f}")
print(f"Flip com Cache     : R$ {history[4][1]:,.2f}")
print(f"Shuffle final      : R$ {history[5][1]:,.2f}")
print(f"Simulated Annealing: R$ {history[6][1]:,.2f}")