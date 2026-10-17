import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres 
N = 5       # nombre de masses
m = 1.0     #masse de chacune des masses
k = 1.0     #coeff de Hook

u = np.zeros(N)     # liste contenant les déplacements des N masses
u[0] = 0.1      # déplacement de la première masse (pour engendrer le mouvement des autres masses)

v = np.zeros(N)     # liste contenant les vitesses des N masses

Y = np.concatenate((u, v))      # (pour utiliser la fonction solve)

def acceleration(n, u):
    u_ext = np.concatenate(([u[-1]], u, [u[0]]))        # pour que la première masse soit reliée à la dernière

    a = (k / m) * (u_ext[n+2] - 2*u_ext[n+1] + u_ext[n])        # décalage de 1 vu qu'on a ajouté u[-1] au début

    return a


def equa_mvmt_chaine(t, Y):
    u = Y[:N]
    v = Y[N:]

    a = np.zeros(N)

    for n in range(N):
        a[n] = acceleration(n, u)

    return np.concatenate((v, a))


t0 = 0.0
tf = 50.0
temps = np.linspace(t0, tf, 2000)

solution = solve_ivp(
    equa_mvmt_chaine,
    (t0, tf),
    Y,
    t_eval=temps
)


U = solution.y[:N]      #récupère les déplacements des N masses

plt.figure()

for n in range(N):
    plt.plot(solution.t, U[n], label=f"Masse {n+1}")    # trace le déplacement de la masse n à chaque instants

plt.xlabel("Temps (s)")
plt.ylabel("Déplacement (m)")
plt.title("Chaine harmonique de ressorts")
plt.legend()
plt.grid()
plt.show()
