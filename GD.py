import numpy as np
import matplotlib.pyplot as plt

# 1. Generación de Datos Sintéticos (Simulamos una tendencia demográfica, por ejemplo)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # La fórmula real es y = 4 + 3x + ruido

# 2. Configuración del Algoritmo
learning_rate = 0.1
n_iterations = 1000
m = len(X) # Número de ejemplos

# Inicialización aleatoria de pesos (theta)
theta = np.random.randn(2, 1)

# Agregamos x0 = 1 a cada instancia para el término de intercepción (bias)
X_b = np.c_[np.ones((m, 1)), X]

# 3. El Bucle del Descenso de Gradiente
cost_history = []

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # Derivada parcial de la función de costo (MSE)
    theta = theta - learning_rate * gradients       # Paso de actualización de la funcion de costo
    
    # Calculamos el costo para ver cómo disminuye:
    cost = np.mean((X_b.dot(theta) - y) ** 2)
    cost_history.append(cost)

print(f"Theta encontrado (Intercepción, Pendiente): \n{theta}")
print(f"Valores reales esperados: Intercepción=4, Pendiente=3")

# 4. Visualización
plt.plot(cost_history)
plt.xlabel('Iteraciones')
plt.ylabel('Costo (MSE)')
plt.title('Convergencia del Descenso de Gradiente')
plt.show()