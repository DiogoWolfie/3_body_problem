"""
Problema dos 3 corpos:
Esse é um gradne problema da física atual. O problema dos 3 corpos consiste basicamente na dificuldade
em prever a trajetória de 3 corpos que se atraem mutuamente, orbitando assim um centro de massa, que muda
com o tempo.
Um exemplo comum é o próprio sistema solar. Se olharmos para a terra, a lua e o sol, eles se comportam como
3 corpos exercendo atração gravitacional uns nos outros. 
Essa será uma tentativa de fazer uma pequnea simulação das órbitas desses corpos. Vai ser bem difícil, já que 
não exite uma solução matemática para isso, tal qual usamo uma elípse para calcular a trajetória de apenas 
2 corpos.
Mas o objetivo é me desafiar, revisar um tequinho de físia, modelagem e cálculo. 
No primeiro momento, tratarei como um problema de 2 dimensões.

Como as forças gravitacionais são conservativas, a energia mecânica total do sistema E 
(que é a soma da energia cinética com energia potencial gravitacional) é constante no tempo. 
Se E < 0, teremos trajetórias elípticas (ou circular, se a excentricidade da elipse for zero); 
se E = 0, ela será parabólica e se E > 0, ela será hiperbólica.

Verlet / Leapfrog
o sistema precisa conservar energia, caso contrário os planeas vão ou se cochar no centro de massa ou ir para infinito.
Para conservar energia, precisamos atualizar velocidade e posição de maneira "intercalada".
No Euler, você faz algo como:

pega a aceleração agora
atualiza tudo usando só esse valor
pronto

Isso cria um erro sistemático: a força muda durante o passo, mas o método finge que ela ficou constante. Em órbitas, 
esse erro vai acumulando e a energia deriva.

No Verlet, você faz:

atualiza a posição usando velocidade e aceleração atuais
recalcula a aceleração na posição nova
corrige a velocidade usando a aceleração antiga e a nova

Isso usa informação dos dois lados do intervalo, então o método fica muito mais estável e tende a preservar 
melhor a energia.


"""


# -- imports --
import numpy as np
from math import *
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -- variáveis -- 
# massaT = 5.972e24  #kg - Terra
# massaM = 6.39e23 # marte
# massaV = 4.87e24# vênus
G = 6.674e-11#constante gravitacional - N.m^2/kg^2
#posições iniciais - decidi fazer um sistema caótico mesmo
# pos_T = np.array([0.0, 1.5e11])     # Terra ~ distância do Sol
# pos_M = np.array([2.2e11, 0.0])
# pos_V = np.array([1.1e11, 0.0])

#velocidades inciais
# v_T = np.array([2000.0, 0.0])
# v_M = np.array([0.0, 1500.0])
# v_V = np.array([0.0, -2500.0])
massaT = massaM = massaV = 1e26  # massas iguais

pos_T = np.array([ 0.97000436e11, -0.24308753e11])
pos_M = np.array([-0.97000436e11,  0.24308753e11])
pos_V = np.array([0.0, 1.6788e11])

v_T = np.array([0.0, 0.0]) 
v_M = np.array([0.0, 0.0])
v_V = np.array([0.0, 0.0])

dt = 10000 #s

traj_T = []
traj_M = []
traj_V = []



escala = 1e11  # 100 bilhões de metros

# -- funções auxiliares --
def forca_gravitacional(m1, m2, pos1, pos2):

    #vetor deslocamento
    vetor = pos2 - pos1 #lembrando que a ordem importa visto que é vetor
    #distancia euclidiana - norma do vetor
    distancia = np.linalg.norm(vetor)
    if distancia == 0:
        return np.array([0.0, 0.0])
    
    epsilon = 1e9  # softening — evita força infinita no encontro próximo
    distancia_suave = np.sqrt(distancia**2 + epsilon**2)
    #vetor unitário
    v_unitario = vetor / distancia_suave
    
    forca_magnitude = m1*m2*G / distancia_suave**2
    forca_vetorial = forca_magnitude * v_unitario

    return forca_vetorial 

def aceleracoes(pos_T,pos_M,pos_V):

    F_T = forca_gravitacional(massaT,massaM,pos_T,pos_M) + forca_gravitacional(massaT, massaV,pos_T,pos_V)
    F_M = forca_gravitacional(massaM,massaT,pos_M,pos_T) + forca_gravitacional(massaM, massaV,pos_M,pos_V)
    F_V = forca_gravitacional(massaV,massaT,pos_V,pos_T) + forca_gravitacional(massaV, massaM,pos_V,pos_M)

    a_T = F_T / massaT
    a_M = F_M / massaM
    a_V = F_V / massaV

    return a_T, a_M, a_V

def centro_de_massa(posicoes, massas):
    massas = np.array(massas, dtype=float)
    posicoes = np.array(posicoes, dtype=float)
    return np.sum(posicoes * massas[:, None], axis=0) / np.sum(massas)


# acelerações iniciais
a_T, a_M, a_V = aceleracoes(pos_T, pos_M, pos_V)

#animação update
def update(frame):
    global pos_T, pos_M, pos_V
    global v_T, v_M, v_V
    global a_T, a_M, a_V

    for _ in range(500):
        #acelerações antigas
        aT_old = a_T.copy()
        aM_old = a_M.copy()
        aV_old = a_V.copy()

        #novas posições -- usando velocity Verlet
        pos_T += v_T * dt + 0.5*a_T*dt**2
        pos_M += v_M * dt + 0.5*a_M*dt**2
        pos_V += v_V * dt + 0.5*a_V*dt**2

        #Novas acelerações
        a_T, a_M, a_V = aceleracoes(pos_T, pos_M, pos_V)

        #velocidades
        v_T += 0.5*(a_T + aT_old)*dt
        v_M += 0.5*(a_M + aM_old)*dt
        v_V += 0.5*(a_V + aV_old)*dt

    cm = centro_de_massa([pos_T, pos_M, pos_V], [massaT, massaM, massaV])

    # posições relativas
    T_rel = (pos_T - cm) / escala
    M_rel = (pos_M - cm) / escala
    V_rel = (pos_V - cm) / escala

    # Salva nas trajetórias
    traj_T.append(T_rel.copy())
    traj_M.append(M_rel.copy())
    traj_V.append(V_rel.copy())

    # Converte pra array e alimenta as linhas
    arr_T = np.array(traj_T)
    arr_M = np.array(traj_M)
    arr_V = np.array(traj_V)

    linha_T.set_data(arr_T[:, 0], arr_T[:, 1])
    linha_M.set_data(arr_M[:, 0], arr_M[:, 1])
    linha_V.set_data(arr_V[:, 0], arr_V[:, 1])

    ponto_T.set_data([T_rel[0]], [T_rel[1]])
    ponto_M.set_data([M_rel[0]], [M_rel[1]])
    ponto_V.set_data([V_rel[0]], [V_rel[1]])

    #camera dinamica
    todas_pos = np.array([T_rel, M_rel, V_rel])
    margem = 0.3

    x_min = todas_pos[:, 0].min() - margem
    x_max = todas_pos[:, 0].max() + margem
    y_min = todas_pos[:, 1].min() - margem
    y_max = todas_pos[:, 1].max() + margem

    # força aspecto igual pra não distorcer as órbitas
    span = max(x_max - x_min, y_max - y_min) / 2
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)

    # Retorna tudo que foi atualizado
    return linha_T, linha_M, linha_V, ponto_T, ponto_M, ponto_V

# -- animação --
fig, ax = plt.subplots(figsize=(8,8))

linha_T, = ax.plot([],[],label = "Terra")
linha_M, = ax.plot([], [], label="Marte")
linha_V, = ax.plot([], [], label="Vênus")

ponto_T, = ax.plot([], [], 'o')
ponto_M, = ax.plot([], [], 'o')
ponto_V, = ax.plot([], [], 'o')

ax.scatter(0, 0, color="black", s=30, label="Centro de massa")

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)

anim = FuncAnimation(fig, update, interval=20)
plt.show()