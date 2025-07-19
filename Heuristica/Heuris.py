import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time

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

capacidade_estadio = 20000
em_casa = capacidade_estadio*0.6
fora_casa = capacidade_estadio*0.4
relevancia_time_1 = 0.9
relevancia_time_2 = 0.6

# 0-terça, 1-quinta, 2-sabado, 3-domingo
dia_semana=[0.6, 0.6, 1, 1.1]
bilheteria = 2
preco_ingresso_base = 52,26

Bilheteria_jogo_time = []
Bilheteria_jogo_time.append(((em_casa * relevancia_time_1) + (fora_casa*relevancia_time_2)))

renda_jogo= Bilheteria_jogo_time[0] * (preco_ingresso_base*dia_semana)
print(Bilheteria_jogo_time[0])

# Etapa 2: geolocalização com fallback
geoloc = Nominatim(user_agent="roundrobin_sched")

def get_coord(city):
    try:
        loc = geoloc.geocode(city, timeout=10)
        if loc:
            time.sleep(1)  # respeita limites da API
            return (loc.latitude, loc.longitude)
        else:
            print(f"Localização não encontrada para: {city}")
            return (0, 0)
    except Exception as e:
        print(f"Erro ao localizar {city}: {e}")
        return (0, 0)

teams['coord'] = teams['city'].apply(get_coord)
teams['longitude'] = teams['coord'].apply(lambda x: x[1])

# Etapa 3: ordena por longitude para logística melhor
teams = teams.sort_values(by='longitude').reset_index(drop=True)

# Etapa 4: custo por km e matriz de custos
C_PER_KM = 0.30
n = len(teams)
cost_mat = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j: continue
        d = geodesic(teams.loc[i, 'coord'], teams.loc[j, 'coord']).km
        cost_mat[i, j] = 2 * d * C_PER_KM  # ida + volta

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

# Etapa 6: controle de casa/fora (corrige sequências longas)
def ajusta_homeaway(schedule, max_away=3):
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

# Etapa 7: otimização simples via troca de rodadas
def swap_and_eval(schedule, iterations=100):
    best = schedule.copy()
    best_cost = cost_of_schedule(best)
    for i in range(iterations):
        new_sched = best.copy()
        # Troca duas rodadas aleatórias
        i1, i2 = np.random.choice(len(schedule), 2, replace=False)
        new_sched[i1], new_sched[i2] = new_sched[i2], new_sched[i1]
        new_cost = cost_of_schedule(new_sched)
        if new_cost < best_cost:
            best = new_sched
            best_cost = new_cost
    return best

# Etapa 8: cálculo do custo total
def cost_of_schedule(schedule):
    total = 0
    for rnd in schedule:
        for a, b in rnd:
            if None in (a, b): continue
            total += cost_mat[a, b]
    return total

# Rodar otimização final
schedule = swap_and_eval(schedule)

# Resultado final
print(f"Custo total estimado: R$ {cost_of_schedule(schedule):,.2f}")
