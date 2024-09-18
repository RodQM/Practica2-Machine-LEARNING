from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def generar_datos_aleatorios(num_puntos, num_caracteristicas):
  
    return np.random.rand(num_puntos, num_caracteristicas) * 10

num_puntos = int(input("¿Cuántos puntos de datos aleatorios deseas generar? "))
num_caracteristicas = int(input("¿Cuántas características (dimensiones) tendrán los puntos? "))

X = generar_datos_aleatorios(num_puntos, num_caracteristicas)

grado = int(input("¿Qué grado de polinomio deseas usar? "))

poly = PolynomialFeatures(degree=grado)

X_poly = poly.fit_transform(X)

print("\nCaracterísticas originales:")
print(X)

nombres_caracteristicas = poly.get_feature_names_out()
print("\nNombres de las características transformadas:")
print(nombres_caracteristicas)

print("\nDatos transformados:")
print(X_poly)
