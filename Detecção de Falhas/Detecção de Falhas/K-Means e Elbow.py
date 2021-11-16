'''------------------ Clusters c/  K-Means -----------------'''

#Importe e pré tratamento de dados

import matplotlib.pyplot as plt
import pandas as pd 

base = pd.read_csv('Sfrio.csv') 
base.describe() 


base_remove1 = base.loc[(base['Cap'] < 10 )]
base = base.drop(base_remove1.index)

base_remove2 = base.loc[(base['Pot'] == 0 )]
base = base.drop(base_remove2.index)

base = base.drop([ 'Vazao', 'Cap', 'Deltad' ],  axis=1)

#K-Means

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


X = base.iloc[:,3:5].values
scaler=StandardScaler()
X= scaler.fit_transform(X)


#Metodo de Elbow

wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, random_state=(0))
    kmeans.fit(X)
    print (i)
    print(kmeans.inertia_)
    wcss.append(kmeans.inertia_) 


plt.plot(range(1, 11), wcss,  palette='tab20')
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()

#Previsões

kmeans = KMeans(n_clusters=3, random_state = 0)
previsoes = kmeans.fit_predict(X)


