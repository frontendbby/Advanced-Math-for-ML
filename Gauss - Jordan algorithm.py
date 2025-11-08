# MÉTODO GAUSS JORDAN - MATEMÁTICAS PARA INGENIERÍA EN INTELIGENCIA ARTIFICIAL
# CORTE 1.

# definir matriz
a = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
n = 3
# algoritmo 1
for k in range(0, n-1):
    for i in range(k+1, n):
        m = a[i][k]/a[k][k]
        for j in range(k, n+1):
            a[i][j] = a[i][j] - m*a[k][j]
for i in range(n):
    for j in range(n+1):
        print(a[i][j], end=' ')

print(a)

# algoritmo 2
x = [0]*n
x[n-1] = a[n-1][n]/a[n-1][n-1]
for i in range(n-1, -1, -1):
    suma = 0
    for j in range(i+1, n):
        suma = suma + a[i][j]*x[j]
    x[i] = (a[i][n]-suma)/a[i][i]
for i in range(n):
    for j in range(n+1):
        print(a[i][j], end=' ')
print(x)


# algoritmo 3
for k in range(0, n):
    # dividir fila k entre a[k][k]
    pivote = a[k][k]
    for j in range(0, n+1):
        a[k][j] = a[k][j]/pivote
    # hacer ceros en columna k
    for i in range(1, n):
        if i != k:
            m = a[i][k]
            for j in range(k, n+1):
                a[i][j] = a[i][j] - m*a[k][j]

print("\n")
