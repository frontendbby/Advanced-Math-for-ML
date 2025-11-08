import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/GerhardSpross/clasificacion-prueba/refs/heads/main/wdbc.data"
df = pd.read_csv(url, header=None)
df.head(10)

# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
LEARNING_RATE = 0.01
ITERATIONS = 50000

# ==============================================================================
# FUNCIONES NÚCLEO (IMPLEMENTACIÓN DESDE CERO)
# ==============================================================================


def sigmoid(z):
    """Función de activación Sigmoid: sigma(z) = 1 / (1 + e^(-z))"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w):
    """Cálculo del Error (Costo) - Binary Cross-Entropy (Log Loss)"""
    m = len(y)
    h = sigmoid(X @ w)
    h = np.clip(h, 1e-15, 1 - 1e-15)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost


def gradient_descent(X, y, w, learning_rate, iterations):
    """Algoritmo de optimización de Gradiente Descendente."""
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X @ w)
        gradient = (1/m) * X.T @ (h - y)
        w -= learning_rate * gradient
    return w


def standardize_features(X):
    """Estandariza las características (media 0, desviación estándar 1)."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma
    return X_norm


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. CARGAR Y PREPROCESAR LOS DATOS
        y_raw = df.iloc[:, 1].to_numpy()
        X_raw = df.iloc[:, 2:].to_numpy().astype(float)
        y = np.where(y_raw == 'M', 1, 0)  # M=1, B=0

        X_norm = standardize_features(X_raw)
        X = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])

        # 2. ENTRENAMIENTO DEL MODELO
        num_features = X.shape[1]
        initial_w = np.zeros(num_features)
        final_w = gradient_descent(X, y, initial_w, LEARNING_RATE, ITERATIONS)

        # 3. CÁLCULO DEL ERROR FINAL
        final_error = compute_cost(X, y, final_w)

        # 4. GENERAR LA SALIDA REQUERIDA (w0 w1 ... wk E)
        output = []
        output.append(f"w0: {final_w[0]:.6f}")
        for i in range(1, num_features):
            output.append(f"w{i}: {final_w[i]:.6f}")
        output.append(f"E: {final_error:.6f}")

        print(' '.join(output))

    except Exception as e:
        print(f"Ocurrió un error durante la ejecución: {e}")
