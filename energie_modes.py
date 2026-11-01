import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Paramètres
N = 16  # nombre de masses
m = 1.0  # masse de chacune des masses
k = 1.0  # coeff de Hook
alpha = 0.25  # coeff de non-linéarité

u = np.zeros(N)  # liste contenant les déplacements des N masses
for i in range(1, N + 1):  # : toute l'energie dans le mode 1
    u[i - 1] = np.sin(i * np.pi / (N + 1))

v = np.zeros(N)  # liste contenant les vitesses des N masses

Y = np.concatenate((u, v))  # (pour utiliser la fonction solve)


def acceleration(n, u):
    u_ext = np.concatenate(
        ([0], u, [0])
    )  # pour que la première masse et la dernière masse soient fixes

    a = (1 / m) * (
        k * (u_ext[n + 2] - 2 * u_ext[n + 1] + u_ext[n])
        + alpha * ((u_ext[n + 2] - u_ext[n + 1]) ** 2 - (u_ext[n + 1] - u_ext[n]) ** 2)
    )  # voir PFD non linéaire idem décalage de 1

    return a


def equa_mvmt_chaine(t, Y):
    u = Y[:N]
    v = Y[N:]

    a = np.zeros(N)

    for n in range(N):
        a[n] = acceleration(n, u)

    return np.concatenate((v, a))


t0 = 0.0
tf = 5000.0
temps = np.linspace(t0, tf, 5000)
nbtemps = len(temps)

solution = solve_ivp(equa_mvmt_chaine, (t0, tf), Y, t_eval=temps)


U = solution.y[:N]  # récupère les déplacements des N masses
V = solution.y[N:]  # récupère les vitesses des N masses


def fourier(T):  # renvoie l'amplitude des k modes propres
    res = np.zeros(N + 1)  # la case 1 correspond au premier mode propre
    for k in range(1, N + 1):
        for i in range(1, N + 1):
            res[k] = res[k] + (2 / (N + 1)) ** 0.5 * T[i - 1] * np.sin(  # Cf latex
                i * k * np.pi / (N + 1)
            )
    return res


omega = np.zeros(N + 1)
for k in range(1, N + 1):  # calcul des pulsations propres
    omega[k] = 2 * np.sin(k * np.pi / (2 * (N + 1)))

# Application de fourier sur les deplacements et les vitesses à chaque instant
FU = np.zeros((N + 1, nbtemps))
FV = np.zeros((N + 1, nbtemps))
for t in range(nbtemps):
    FU[:, t] = fourier(U[:, t])
    FV[:, t] = fourier(V[:, t])

E = np.zeros((N + 1, len(temps)))
for k in range(1, N + 1):
    for t in range(len(temps)):
        E[k][t] = 0.5 * (
            FV[k, t] ** 2 + (omega[k] * FU[k, t]) ** 2
        )  # calcul de l'energie par mode à chaque instant

for n in range(1, N + 1):
    plt.plot(
        solution.t, E[n], label=f"Energie {n}"
    )  # trace le déplacement de l'énergie par mode n à chaque instants

plt.xlabel("Temps (s)")
plt.ylabel("Energie")
plt.title("Énergie des modes propres au cours du temps")
plt.legend()
plt.grid()
plt.show()
