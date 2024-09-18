#Codigo de SCORE

from sklearn.cluster import KMeans
import numpy as np

n = int(input("Ingresa el número de puntos de datos: "))
num_clusters = int(input("Ingresa el número de clústeres: "))


X = np.random.rand(n, 2) * 10  
print("Puntos generados:")
print(X)

if num_clusters > len(X):
    print(f"Error: el número de clústeres ({num_clusters}) no puede ser mayor al número de puntos ({len(X)}).")
else:
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

  
    score = kmeans.score(X)
    print(f'Score: {score}')