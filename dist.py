#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Fijar la semilla
np.random.seed(321)

# Generar datos, sampleados de Bern(theta=0.4)
x = np.random.binomial(n=1, p=0.4, size=100)

# Estimación máximo verosímil
theta_hat = np.mean(x)
print(f"Estimación máximo verosímil (theta_hat): {theta_hat}")

# Estimación de la varianza asintótica
var_theta = theta_hat * (1 - theta_hat)
print(f"Estimación de la varianza asintótica: {var_theta}")

# Estimación del error estándar
se_hat = np.sqrt(var_theta / 100)
print(f"Estimación del error estándar: {se_hat}")

# Error estándar aproximado del estimador
se_aprox = np.sqrt(0.24 / 100)
print(f"Error estándar aproximado del estimador: {se_aprox}")

# Gráfico de la distribución aproximada del EMV
x_vals = np.linspace(0, 1, 100)
y_vals = norm.pdf(x_vals, 0.4, se_aprox)

plt.figure()
plt.plot(x_vals, y_vals, color='blue', linewidth=2)
plt.xlabel('Valores del Estimador')
plt.ylabel('Densidad')
plt.title('Distribución Aproximada del EMV')
plt.show()

# Simulación para comprobar el resultado teórico
estimaciones = []

for i in range(10000):
    x = np.random.binomial(n=1, p=0.4, size=100)
    estimaciones.append(np.mean(x))

# Histograma de las estimaciones con la distribución teórica superpuesta
plt.figure()
plt.hist(estimaciones, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
plt.plot(x_vals, y_vals, color='blue', linewidth=2)
plt.ylim(0, 8)
plt.xlabel('Valores del Estimador')
plt.ylabel('Densidad')
plt.title('Distribución de las Estimaciones Simuladas')
plt.show()
