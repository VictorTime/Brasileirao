import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import matplotlib.pyplot as plt
import networkx as nx

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

#atribuindo aleatoriamente um peso 
rel_dict = {c: np.random.uniform(0.4, 0.9) for c in teams.code}

print(rel_dict["CAP"])
#criando os times usando o que tem la dataframe
teams_objs = [TimeF(nome=row.code, relevancia=rel_dict[row.code], cidade=row.city) for idx,row in teams.iterrows()]

dia_semanas=[0.6, 0.6, 1, 1.1]

# a gente, em tese, tem que fazer o capacidade do estadio variar
estadio = {t.nome: Estadio(20000, t.cidade, dias_semana=dia_semanas) for t in teams_objs}


# Etapa 2: geolocalização com fallback
geoloc = Nominatim(user_agent="roundrobin_sched")
coords = {}
for t in teams_objs:
    loc = geoloc.geocode(t.cidade, timeout=10)
    time.sleep(1)
    coords[t.nome] = (loc.latitude, loc.longitude) if loc else (0,0)


# Etapa 4: custo por km e matriz de custos
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
    ids = list(team_ids)
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
        dia = r_idx % len(dia_semanas)
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

def heuristic_greedy_swap(schedule, iters=200):
    best, best_profit = schedule, lucro_schedule(schedule)
    for _ in range(iters):
        new = [r[:] for r in best]
        i, j = np.random.choice(len(new), 2, replace=False)
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

# ——— Pipeline de otimização —————————————————————

history = []
current, cp = schedule, lucro_schedule(schedule)
history.append(('Inicial', cp))

current, p1 = heuristic_greedy_swap(current, 200)
history.append(('Swap', p1))
current, p2 = heuristic_flip_homeaway(current, 200)
history.append(('Flip', p2))
current, p3 = heuristic_restrict_shuffle(current, 4, 100)
history.append(('Shuffle', p3))

# ——— Plot de evolução do lucro ————————————————————

steps, profits = zip(*history)
plt.figure(figsize=(8,5))
plt.plot(steps, profits, marker='o', linestyle='-', color='blue')
plt.title('Evolução do Lucro Durante Otimização')
plt.xlabel('Etapa')
plt.ylabel('Lucro Estimado (R$)')
plt.grid(True)
for i, (x,y) in enumerate(zip(steps, profits)):
    plt.annotate(f"R$ {y:,.0f}", (i,y), textcoords="offset points",
                 xytext=(0,8), ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))  # exemplo de anotação 
plt.tight_layout()
plt.show()