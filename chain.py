import sympy as sp
from sympy import Matrix

x = sp.symbols('x')

funcion = (3*x + 2)**2
derivada = sp.diff(funcion, x)

print(f" La funcion original es: {funcion}")
print(f"La derivada de la funcion calculada por Python es: {derivada}")

print(f"La derivada expandida es: {sp.expand(derivada)}")