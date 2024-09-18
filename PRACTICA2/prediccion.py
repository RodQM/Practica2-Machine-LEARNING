# Importar las librerías necesarias
from sklearn.cluster import KMeans
import numpy as np

# Función para generar puntos aleatorios
def generar_datos_aleatorios(num_puntos, num_caracteristicas):
    # Generar datos aleatorios entre 0 y 10
    return np.random.rand(num_puntos, num_caracteristicas) * 10

# Función para obtener nuevas muestras del usuario
def obtener_nuevas_muestras():
    n = int(input("¿Cuántas nuevas muestras deseas predecir? "))
    muestras = []
    for i in range(n):
        muestra = list(map(float, input(f"Ingrese las coordenadas de la muestra {i + 1} (separadas por espacio): ").split()))
        muestras.append(muestra)
    return np.array(muestras)

# Parámetros para los datos aleatorios
num_puntos = int(input("¿Cuántos puntos de datos aleatorios deseas generar? "))
num_caracteristicas = int(input("¿Cuántas características (dimensiones) tendrán los puntos? "))

# Generar datos aleatorios
X = generar_datos_aleatorios(num_puntos, num_caracteristicas)

# Mostrar los puntos generados
print("\nPuntos de datos generados aleatoriamente:")
print(X)

# Preguntar cuántos clústeres desea
n_clusters = int(input("\n¿Cuántos clústeres quieres formar? "))

# Inicializar y ajustar el modelo KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Mostrar los centroides encontrados
print("\nCentroides del modelo:")
print(kmeans.cluster_centers_)

# Obtener nuevas muestras para predecir
print("\nIntroduce nuevas muestras para predecir su clúster:")
nuevas_muestras = obtener_nuevas_muestras()

# Predecir a qué clúster pertenecen las nuevas muestras
labels = kmeans.predict(nuevas_muestras)

print("\nPredicciones de las nuevas muestras (índices de clúster):")
print(labels)
