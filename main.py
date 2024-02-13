import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import matplotlib.pyplot as plt

# Citirea datelor ARFF
data, meta = arff.loadarff(’iris.arff’)

# Transformarea datelor ARFF intr-un DataFrame
df = pd.DataFrame(data)

# Afisarea primelor cateva randuri ale setului de date
print(df.head())

# Selectarea trasaturilor ce le vom folosi pentru clusterizare
features = df[[’sepallength’, ’sepalwidth’, ’petallength’, ’petalwidth’]]

# Standardizarea datelor
scaler = StandardScaler()
scaled features = scaler.fit transform(features)

# Alegerea numarului de clustere
num clusters = 3

# Crearea si aplicarea modelului K-Means
kmeans = KMeans(n clusters=num clusters, random state=42)
df[’Cluster’] = kmeans.fit predict(scaled features)

# Afisarea rezultatelor clusterizarii
print(df[[’Planta’, ’Cluster’]])

# Afisarea centroidelor (valorile medii) ale fiecarui cluster in spatiul trasaturilor standardizate
print(”Centroidele:”)
print(scaler.inverse transform(kmeans.cluster centers ))

# Vizualizarea rezultatelor clusterizarii
plt.scatter(df[’sepallength’], df[’sepalwidth’], c=df[’Cluster’], cmap=’viridis’)
plt.scatter(scaler.inverse transform(kmeans.cluster centers )[:, 0], scaler.inverse transform(kmeans.cluster centers )
[:, 1], marker=’X’, s=200, c=’red’)
plt.xlabel(’Sepal Length’)
plt.ylabel(’Sepal Width’)
plt.title(’Clusterizare Iris cu K-Means’)
plt.show()