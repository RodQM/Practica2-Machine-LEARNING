


from sklearn.cluster import KMeans
import numpy as np


def generar_datos_aleatorios(num_puntos, num_caracteristicas):
    # Generar datos aleatorios entre 0 y 10
    return np.random.rand(num_puntos, num_caracteristicas) * 10

def obtener_nuevas_muestras():
    n = int(input("¿Cuántas nuevas muestras deseas predecir? "))
    muestras = []
    for i in range(n):
        muestra = list(map(float, input(f"Ingrese las coordenadas de la muestra {i + 1} (separadas por espacio): ").split()))
        muestras.append(muestra)
    return np.array(muestras)

num_puntos = int(input("¿Cuántos puntos de datos aleatorios deseas generar? "))
num_caracteristicas = int(input("¿Cuántas características (dimensiones) tendrán los puntos? "))

X = generar_datos_aleatorios(num_puntos, num_caracteristicas)

print("\nPuntos de datos generados aleatoriamente:")
print(X)

n_clusters = int(input("\n¿Cuántos clústeres quieres formar? "))

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

print("\nCentroides del modelo:")
print(kmeans.cluster_centers_)

print("\nIntroduce nuevas muestras para predecir su clúster:")
nuevas_muestras = obtener_nuevas_muestras()

labels = kmeans.predict(nuevas_muestras)

print("\nPredicciones de las nuevas muestras (índices de clúster):")
print(labels)
