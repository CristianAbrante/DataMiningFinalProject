#Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#From
from collections import Counter

#Preprocessing
from sklearn.preprocessing import MinMaxScaler #Min max scaler
from sklearn.preprocessing import StandardScaler #Standard scaler
from sklearn.decomposition import PCA #PCA
from sklearn.decomposition import TruncatedSVD #SVD

#Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import dbscan

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

#Scores for the number of clusters
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score

#Main
if __name__ == "__main__":
	#Read df
	df=pd.read_csv('msdata.csv',na_values=['?','#','&','@'],keep_default_na=True)
	
	classes=np.array(df.pop('class'))-1 #Pop the classes
	ids=df.pop('id') #Pop the id
	cols=df.columns #Save columns
	
	
	#Time to edit the data, scaling variables
	sdscaler = StandardScaler()
	df_scale=pd.DataFrame(sdscaler.fit_transform(df),columns=cols)
	
	pca1=PCA(n_components=2) #2 components
	pca2=PCA(n_components=3) #3 Components
	pca3=PCA(n_components=4) #4 Components
	pca4=PCA(n_components=.5) #50%Variance
	pca5=PCA(n_components=.6) #60%Variance
	pca6=PCA(n_components=.7) #70%Variance
	pca7=PCA(n_components=.8) #80%Variance
	pca8=PCA(n_components=.9) #90%Variance


	df_pca1 = pd.DataFrame(pca1.fit_transform(df_scale))
	df_pca2 = pd.DataFrame(pca2.fit_transform(df_scale))
	df_pca3 = pd.DataFrame(pca3.fit_transform(df_scale))
	df_pca4 = pd.DataFrame(pca4.fit_transform(df_scale))
	df_pca5 = pd.DataFrame(pca5.fit_transform(df_scale))
	df_pca6 = pd.DataFrame(pca6.fit_transform(df_scale))
	df_pca7 = pd.DataFrame(pca7.fit_transform(df_scale))
	df_pca8 = pd.DataFrame(pca8.fit_transform(df_scale))
	
	#[('Agglomerative clustering 3 Clusters, complete linkage, Cosine, 2 Components: ', 0.8704595404687339), ('Agglomerative clustering 6 Clusters, Single linkage, Cosine, 2 Components: ', 0.8527480849011208), ('Agglomerative clustering 3 Clusters, average linkage, Cosine, 4 Components: ', 0.8308527272121102), ('Agglomerative clustering 7 Clusters, Single linkage, Cosine, 2 Components: ', 0.8276498843162884), ('Agglomerative clustering 3 Clusters, average linkage, Cosine, 2 Components: ', 0.8187407994276206), ('Agglomerative clustering 4 Clusters, average linkage, Cosine, 2 Components: ', 0.8151717946668879), ('Agglomerative clustering 4 Clusters, complete linkage, Cosine, 2 Components: ', 0.7909766331843715), ('Agglomerative clustering 5 Clusters, average linkage, Cosine, 2 Components: ', 0.7860058411489088), ('Kmeans 5 Clusters, 2 Components: ', 0.7812258336277624), ('Agglomerative clustering 3 Clusters, complete linkage, Cosine, 4 Components: ', 0.7740622345653174)]

	
	#Choose what to run
	kmeans=True
	
	aggc_euclidean_single=True
	aggc_l1_single=True
	aggc_l2_single=True
	aggc_manhattan_single=True
	aggc_cosine_single=True
	
	aggc_euclidean_complete=True
	aggc_l1_complete=True
	aggc_l2_complete=True
	aggc_manhattan_complete=True
	aggc_cosine_complete=True
	
	aggc_euclidean_average=True
	aggc_l1_average=True
	aggc_l2_average=True
	aggc_manhattan_average=True
	aggc_cosine_average=True
	
	aggc_centroid=True
	
	#These are giving trouble
	spec_clus_gamma05=True
	spec_clus_gamma10=True
	spec_clus_gamma15=True
	spec_clus_gamma20=True
	
	results=[]
	
	if kmeans:
		print('#################Kmeans#################')
		
		kmeans1 = KMeans(n_clusters=3)
		kmeans2 = KMeans(n_clusters=4)
		kmeans3 = KMeans(n_clusters=5)
		kmeans4 = KMeans(n_clusters=6)
		kmeans5 = KMeans(n_clusters=7)
	
		#Kmeans 2 PCA components
		kmeans1.fit(df_pca1)
		results.append(("Kmeans 3 Clusters, 2 Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca1)
		results.append(("Kmeans 4 Clusters, 2 Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca1)
		results.append(("Kmeans 5 Clusters, 2 Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca1)
		results.append(("Kmeans 6 Clusters, 2 Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca1)
		results.append(("Kmeans 7 Clusters, 2 Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 3 PCA components
		kmeans1.fit(df_pca2)
		results.append(("Kmeans 3 Clusters, 3 Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca2)
		results.append(("Kmeans 4 Clusters, 3 Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca2)
		results.append(("Kmeans 5 Clusters, 3 Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca2)
		results.append(("Kmeans 6 Clusters, 3 Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca2)
		results.append(("Kmeans 7 Clusters, 3 Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 4 PCA components
		kmeans1.fit(df_pca3)
		results.append(("Kmeans 3 Clusters, 4 Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca3)
		results.append(("Kmeans 4 Clusters, 4 Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca3)
		results.append(("Kmeans 5 Clusters, 4 Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca3)
		results.append(("Kmeans 6 Clusters, 4 Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca3)
		results.append(("Kmeans 7 Clusters, 4 Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 50% PCA components
		kmeans1.fit(df_pca4)
		results.append(("Kmeans 3 Clusters, 50% Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca4)
		results.append(("Kmeans 4 Clusters, 50% Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca4)
		results.append(("Kmeans 5 Clusters, 50% Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca4)
		results.append(("Kmeans 6 Clusters, 50% Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca4)
		results.append(("Kmeans 7 Clusters, 50% Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 60% PCA components
		kmeans1.fit(df_pca5)
		results.append(("Kmeans 3 Clusters, 60% Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca5)
		results.append(("Kmeans 4 Clusters, 60% Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca5)
		results.append(("Kmeans 5 Clusters, 60% Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca5)
		results.append(("Kmeans 6 Clusters, 60% Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca5)
		results.append(("Kmeans 7 Clusters, 60% Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 70% PCA components
		kmeans1.fit(df_pca6)
		results.append(("Kmeans 3 Clusters, 70% Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca6)
		results.append(("Kmeans 4 Clusters, 70% Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca6)
		results.append(("Kmeans 5 Clusters, 70% Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca6)
		results.append(("Kmeans 6 Clusters, 70% Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca6)
		results.append(("Kmeans 7 Clusters, 70% Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 80% PCA components
		kmeans1.fit(df_pca7)
		results.append(("Kmeans 3 Clusters, 80% Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca7)
		results.append(("Kmeans 4 Clusters, 80% Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca7)
		results.append(("Kmeans 5 Clusters, 80% Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca7)
		results.append(("Kmeans 6 Clusters, 80% Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca7)
		results.append(("Kmeans 7 Clusters, 80% Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
		
		#Kmeans 90% PCA components
		kmeans1.fit(df_pca8)
		results.append(("Kmeans 3 Clusters, 90% Components: ",normalized_mutual_info_score(classes,kmeans1.labels_,average_method='geometric')))
		kmeans2.fit(df_pca8)
		results.append(("Kmeans 4 Clusters, 90% Components: ",normalized_mutual_info_score(classes,kmeans2.labels_,average_method='geometric')))
		kmeans3.fit(df_pca8)
		results.append(("Kmeans 5 Clusters, 90% Components: ",normalized_mutual_info_score(classes,kmeans3.labels_,average_method='geometric')))
		kmeans4.fit(df_pca8)
		results.append(("Kmeans 6 Clusters, 90% Components: ",normalized_mutual_info_score(classes,kmeans4.labels_,average_method='geometric')))
		kmeans5.fit(df_pca8)
		results.append(("Kmeans 7 Clusters, 90% Components: ",normalized_mutual_info_score(classes,kmeans5.labels_,average_method='geometric')))
	
	if aggc_euclidean_single:
		#Agglomerative clustering euclidian Single
		print('#################Agglomerative Clustering Euclidean Single#################')
		
		#Single-Euclidean
		aggc_single3 = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')
		aggc_single4 = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')
		aggc_single5 = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='single')
		aggc_single6 = AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='single')
		aggc_single7 = AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='single')
	
		#Agglomerative Single Euclidean 2 PCA components
		aggc_single3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 3 PCA components
		aggc_single3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 4 PCA components
		aggc_single3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 50% PCA components
		aggc_single3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 60% PCA components
		aggc_single3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 70% PCA components
		aggc_single3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 80% PCA components
		aggc_single3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Euclidean 90% PCA components
		aggc_single3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
	
	if aggc_l1_single:
		
		#Agglomerative clustering L1 Single
		print('#################Agglomerative Clustering L1 Single#################')
		
		#Single-L1
		aggc_single3 = AgglomerativeClustering(n_clusters=3,affinity='l1',linkage='single')
		aggc_single4 = AgglomerativeClustering(n_clusters=4,affinity='l1',linkage='single')
		aggc_single5 = AgglomerativeClustering(n_clusters=5,affinity='l1',linkage='single')
		aggc_single6 = AgglomerativeClustering(n_clusters=6,affinity='l1',linkage='single')
		aggc_single7 = AgglomerativeClustering(n_clusters=7,affinity='l1',linkage='single')
		
		#Agglomerative Single L1 2 PCA components
		aggc_single3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 3 PCA components
		aggc_single3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 4 PCA components
		aggc_single3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 50% PCA components
		aggc_single3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 60% PCA components
		aggc_single3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 70% PCA components
		aggc_single3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 80% PCA components
		aggc_single3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L1 90% PCA components
		aggc_single3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
	
	if aggc_l2_single:
		#Agglomerative clustering L2 Single
		print('#################Agglomerative Clustering L2 Single#################')
		
		#Single-L2
		aggc_single3 = AgglomerativeClustering(n_clusters=3,affinity='l2',linkage='single')
		aggc_single4 = AgglomerativeClustering(n_clusters=4,affinity='l2',linkage='single')
		aggc_single5 = AgglomerativeClustering(n_clusters=5,affinity='l2',linkage='single')
		aggc_single6 = AgglomerativeClustering(n_clusters=6,affinity='l2',linkage='single')
		aggc_single7 = AgglomerativeClustering(n_clusters=7,affinity='l2',linkage='single')
		
		#Agglomerative Single L2 2 PCA components
		aggc_single3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 3 PCA components
		aggc_single3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 4 PCA components
		aggc_single3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 50% PCA components
		aggc_single3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 60% PCA components
		aggc_single3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 70% PCA components
		aggc_single3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 80% PCA components
		aggc_single3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single L2 90% PCA components
		aggc_single3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
	
	if aggc_manhattan_single:
		#Agglomerative clustering Manhattan Single
		print('#################Agglomerative Clustering Manhattan Single#################')
		
		#Single-Manhattan
		aggc_single3 = AgglomerativeClustering(n_clusters=3,affinity='manhattan',linkage='single')
		aggc_single4 = AgglomerativeClustering(n_clusters=4,affinity='manhattan',linkage='single')
		aggc_single5 = AgglomerativeClustering(n_clusters=5,affinity='manhattan',linkage='single')
		aggc_single6 = AgglomerativeClustering(n_clusters=6,affinity='manhattan',linkage='single')
		aggc_single7 = AgglomerativeClustering(n_clusters=7,affinity='manhattan',linkage='single')
		
		#Agglomerative Single Manhattan 2 PCA components
		aggc_single3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 3 PCA components
		aggc_single3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 4 PCA components
		aggc_single3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 50% PCA components
		aggc_single3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 60% PCA components
		aggc_single3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 70% PCA components
		aggc_single3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 80% PCA components
		aggc_single3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Manhattan 90% PCA components
		aggc_single3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))	
	
	if aggc_cosine_single:
		#Agglomerative clustering Cosine Single
		print('#################Agglomerative Clustering Cosine Single#################')
		
		#Single-Cosine
		aggc_single3 = AgglomerativeClustering(n_clusters=3,affinity='cosine',linkage='single')
		aggc_single4 = AgglomerativeClustering(n_clusters=4,affinity='cosine',linkage='single')
		aggc_single5 = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='single')
		aggc_single6 = AgglomerativeClustering(n_clusters=6,affinity='cosine',linkage='single')
		aggc_single7 = AgglomerativeClustering(n_clusters=7,affinity='cosine',linkage='single')
		
		#Agglomerative Single Cosine 2 PCA components
		aggc_single3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 3 PCA components
		aggc_single3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 4 PCA components
		aggc_single3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 50% PCA components
		aggc_single3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 60% PCA components
		aggc_single3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 70% PCA components
		aggc_single3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 80% PCA components
		aggc_single3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))
		
		#Agglomerative Single Cosine 90% PCA components
		aggc_single3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, Single linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_single3.labels_,average_method='geometric')))
		aggc_single4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, Single linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_single4.labels_,average_method='geometric')))
		aggc_single5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, Single linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_single5.labels_,average_method='geometric')))
		aggc_single6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_single6.labels_,average_method='geometric')))
		aggc_single7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, Single linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_single7.labels_,average_method='geometric')))	
	
	if aggc_euclidean_complete:
		#Agglomerative clustering euclidian complete
		print('#################Agglomerative Clustering Euclidean Complete#################')
		
		#complete-Euclidean
		aggc_complete3 = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
		aggc_complete4 = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')
		aggc_complete5 = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
		aggc_complete6 = AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='complete')
		aggc_complete7 = AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='complete')
	
		#Agglomerative Complete Euclidean 2 PCA components
		aggc_complete3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 3 PCA components
		aggc_complete3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 4 PCA components
		aggc_complete3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 50% PCA components
		aggc_complete3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 60% PCA components
		aggc_complete3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 70% PCA components
		aggc_complete3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 80% PCA components
		aggc_complete3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Euclidean 90% PCA components
		aggc_complete3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
	
	if aggc_l1_complete:
		
		#Agglomerative clustering L1 complete
		print('#################Agglomerative Clustering L1 complete#################')
		
		#complete-L1
		aggc_complete3 = AgglomerativeClustering(n_clusters=3,affinity='l1',linkage='complete')
		aggc_complete4 = AgglomerativeClustering(n_clusters=4,affinity='l1',linkage='complete')
		aggc_complete5 = AgglomerativeClustering(n_clusters=5,affinity='l1',linkage='complete')
		aggc_complete6 = AgglomerativeClustering(n_clusters=6,affinity='l1',linkage='complete')
		aggc_complete7 = AgglomerativeClustering(n_clusters=7,affinity='l1',linkage='complete')
		
		#Agglomerative complete L1 2 PCA components
		aggc_complete3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 3 PCA components
		aggc_complete3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 4 PCA components
		aggc_complete3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 50% PCA components
		aggc_complete3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 60% PCA components
		aggc_complete3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 70% PCA components
		aggc_complete3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 80% PCA components
		aggc_complete3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L1 90% PCA components
		aggc_complete3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
	
	if aggc_l2_complete:
		#Agglomerative clustering L2 complete
		print('#################Agglomerative Clustering L2 complete#################')
		
		#complete-L2
		aggc_complete3 = AgglomerativeClustering(n_clusters=3,affinity='l2',linkage='complete')
		aggc_complete4 = AgglomerativeClustering(n_clusters=4,affinity='l2',linkage='complete')
		aggc_complete5 = AgglomerativeClustering(n_clusters=5,affinity='l2',linkage='complete')
		aggc_complete6 = AgglomerativeClustering(n_clusters=6,affinity='l2',linkage='complete')
		aggc_complete7 = AgglomerativeClustering(n_clusters=7,affinity='l2',linkage='complete')
		
		#Agglomerative complete L2 2 PCA components
		aggc_complete3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 3 PCA components
		aggc_complete3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 4 PCA components
		aggc_complete3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 50% PCA components
		aggc_complete3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 60% PCA components
		aggc_complete3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 70% PCA components
		aggc_complete3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 80% PCA components
		aggc_complete3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete L2 90% PCA components
		aggc_complete3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
	
	if aggc_manhattan_complete:
		#Agglomerative clustering Manhattan complete
		print('#################Agglomerative Clustering Manhattan complete#################')
		
		#complete-Manhattan
		aggc_complete3 = AgglomerativeClustering(n_clusters=3,affinity='manhattan',linkage='complete')
		aggc_complete4 = AgglomerativeClustering(n_clusters=4,affinity='manhattan',linkage='complete')
		aggc_complete5 = AgglomerativeClustering(n_clusters=5,affinity='manhattan',linkage='complete')
		aggc_complete6 = AgglomerativeClustering(n_clusters=6,affinity='manhattan',linkage='complete')
		aggc_complete7 = AgglomerativeClustering(n_clusters=7,affinity='manhattan',linkage='complete')
		
		#Agglomerative complete Manhattan 2 PCA components
		aggc_complete3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 3 PCA components
		aggc_complete3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 4 PCA components
		aggc_complete3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 50% PCA components
		aggc_complete3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 60% PCA components
		aggc_complete3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 70% PCA components
		aggc_complete3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 80% PCA components
		aggc_complete3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Manhattan 90% PCA components
		aggc_complete3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))	
	
	if aggc_cosine_complete:
		#Agglomerative clustering Cosine complete
		print('#################Agglomerative Clustering Cosine complete#################')
		
		#complete-Cosine
		aggc_complete3 = AgglomerativeClustering(n_clusters=3,affinity='cosine',linkage='complete')
		aggc_complete4 = AgglomerativeClustering(n_clusters=4,affinity='cosine',linkage='complete')
		aggc_complete5 = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='complete')
		aggc_complete6 = AgglomerativeClustering(n_clusters=6,affinity='cosine',linkage='complete')
		aggc_complete7 = AgglomerativeClustering(n_clusters=7,affinity='cosine',linkage='complete')
		
		#Agglomerative complete Cosine 2 PCA components
		aggc_complete3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 3 PCA components
		aggc_complete3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 4 PCA components
		aggc_complete3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 50% PCA components
		aggc_complete3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 60% PCA components
		aggc_complete3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 70% PCA components
		aggc_complete3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 80% PCA components
		aggc_complete3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))
		
		#Agglomerative complete Cosine 90% PCA components
		aggc_complete3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, complete linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete3.labels_,average_method='geometric')))
		aggc_complete4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, complete linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete4.labels_,average_method='geometric')))
		aggc_complete5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, complete linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete5.labels_,average_method='geometric')))
		aggc_complete6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, complete linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete6.labels_,average_method='geometric')))
		aggc_complete7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, complete linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_complete7.labels_,average_method='geometric')))	
	
	if aggc_euclidean_average:
		#Agglomerative clustering euclidian average
		print('#################Agglomerative Clustering Euclidean average#################')
		
		#average-Euclidean
		aggc_average3 = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='average')
		aggc_average4 = AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='average')
		aggc_average5 = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='average')
		aggc_average6 = AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='average')
		aggc_average7 = AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='average')
	
		#Agglomerative average Euclidean 2 PCA components
		aggc_average3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 2 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 3 PCA components
		aggc_average3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 3 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 4 PCA components
		aggc_average3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 4 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 50% PCA components
		aggc_average3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 50% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 60% PCA components
		aggc_average3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 60% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 70% PCA components
		aggc_average3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 70% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 80% PCA components
		aggc_average3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 80% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Euclidean 90% PCA components
		aggc_average3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Euclidean, 90% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
	
	if aggc_l1_average:
		
		#Agglomerative clustering L1 average
		print('#################Agglomerative Clustering L1 average#################')
		
		#average-L1
		aggc_average3 = AgglomerativeClustering(n_clusters=3,affinity='l1',linkage='average')
		aggc_average4 = AgglomerativeClustering(n_clusters=4,affinity='l1',linkage='average')
		aggc_average5 = AgglomerativeClustering(n_clusters=5,affinity='l1',linkage='average')
		aggc_average6 = AgglomerativeClustering(n_clusters=6,affinity='l1',linkage='average')
		aggc_average7 = AgglomerativeClustering(n_clusters=7,affinity='l1',linkage='average')
		
		#Agglomerative average L1 2 PCA components
		aggc_average3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 2 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 3 PCA components
		aggc_average3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 3 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 4 PCA components
		aggc_average3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 4 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 50% PCA components
		aggc_average3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 50% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 60% PCA components
		aggc_average3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 60% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 70% PCA components
		aggc_average3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 70% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 80% PCA components
		aggc_average3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 80% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L1 90% PCA components
		aggc_average3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L1, 90% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
	
	if aggc_l2_average:
		#Agglomerative clustering L2 average
		print('#################Agglomerative Clustering L2 average#################')
		
		#average-L2
		aggc_average3 = AgglomerativeClustering(n_clusters=3,affinity='l2',linkage='average')
		aggc_average4 = AgglomerativeClustering(n_clusters=4,affinity='l2',linkage='average')
		aggc_average5 = AgglomerativeClustering(n_clusters=5,affinity='l2',linkage='average')
		aggc_average6 = AgglomerativeClustering(n_clusters=6,affinity='l2',linkage='average')
		aggc_average7 = AgglomerativeClustering(n_clusters=7,affinity='l2',linkage='average')
		
		#Agglomerative average L2 2 PCA components
		aggc_average3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 2 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 3 PCA components
		aggc_average3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 3 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 4 PCA components
		aggc_average3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 4 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 50% PCA components
		aggc_average3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 50% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 60% PCA components
		aggc_average3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 60% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 70% PCA components
		aggc_average3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 70% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 80% PCA components
		aggc_average3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 80% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average L2 90% PCA components
		aggc_average3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, L2, 90% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
	
	if aggc_manhattan_average:
		#Agglomerative clustering Manhattan average
		print('#################Agglomerative Clustering Manhattan average#################')
		
		#average-Manhattan
		aggc_average3 = AgglomerativeClustering(n_clusters=3,affinity='manhattan',linkage='average')
		aggc_average4 = AgglomerativeClustering(n_clusters=4,affinity='manhattan',linkage='average')
		aggc_average5 = AgglomerativeClustering(n_clusters=5,affinity='manhattan',linkage='average')
		aggc_average6 = AgglomerativeClustering(n_clusters=6,affinity='manhattan',linkage='average')
		aggc_average7 = AgglomerativeClustering(n_clusters=7,affinity='manhattan',linkage='average')
		
		#Agglomerative average Manhattan 2 PCA components
		aggc_average3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 2 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 3 PCA components
		aggc_average3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 3 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 4 PCA components
		aggc_average3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 4 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 50% PCA components
		aggc_average3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 50% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 60% PCA components
		aggc_average3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 60% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 70% PCA components
		aggc_average3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 70% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 80% PCA components
		aggc_average3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 80% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Manhattan 90% PCA components
		aggc_average3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Manhattan, 90% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))	
	
	if aggc_cosine_average:
		#Agglomerative clustering Cosine average
		print('#################Agglomerative Clustering Cosine average#################')
		
		#average-Cosine
		aggc_average3 = AgglomerativeClustering(n_clusters=3,affinity='cosine',linkage='average')
		aggc_average4 = AgglomerativeClustering(n_clusters=4,affinity='cosine',linkage='average')
		aggc_average5 = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='average')
		aggc_average6 = AgglomerativeClustering(n_clusters=6,affinity='cosine',linkage='average')
		aggc_average7 = AgglomerativeClustering(n_clusters=7,affinity='cosine',linkage='average')
		
		#Agglomerative average Cosine 2 PCA components
		aggc_average3.fit(df_pca1)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca1)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca1)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca1)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca1)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 2 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 3 PCA components
		aggc_average3.fit(df_pca2)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca2)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca2)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca2)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca2)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 3 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 4 PCA components
		aggc_average3.fit(df_pca3)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca3)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca3)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca3)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca3)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 4 Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 50% PCA components
		aggc_average3.fit(df_pca4)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca4)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca4)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca4)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca4)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 50% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 60% PCA components
		aggc_average3.fit(df_pca5)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca5)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca5)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca5)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca5)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 60% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 70% PCA components
		aggc_average3.fit(df_pca6)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca6)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca6)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca6)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca6)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 70% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 80% PCA components
		aggc_average3.fit(df_pca7)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca7)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca7)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca7)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca7)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 80% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))
		
		#Agglomerative average Cosine 90% PCA components
		aggc_average3.fit(df_pca8)
		results.append(("Agglomerative clustering 3 Clusters, average linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_average3.labels_,average_method='geometric')))
		aggc_average4.fit(df_pca8)
		results.append(("Agglomerative clustering 4 Clusters, average linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_average4.labels_,average_method='geometric')))
		aggc_average5.fit(df_pca8)
		results.append(("Agglomerative clustering 5 Clusters, average linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_average5.labels_,average_method='geometric')))
		aggc_average6.fit(df_pca8)
		results.append(("Agglomerative clustering 6 Clusters, average linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_average6.labels_,average_method='geometric')))
		aggc_average7.fit(df_pca8)
		results.append(("Agglomerative clustering 7 Clusters, average linkage, Cosine, 90% Components: ",normalized_mutual_info_score(classes,aggc_average7.labels_,average_method='geometric')))	
	
	if aggc_centroid:
		#Agglomerative clustering centroid
		print('#################Agglomerative Clustering Centroid#################')
		linkage_centroid1 = linkage(df_pca1, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid1[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid1 = fcluster(linkage_centroid1,t=thresh)
		results.append(("Agglomerative clustering Centroid 2 Components: ",normalized_mutual_info_score(classes,aggc_centroid1,average_method='geometric')))

		
		linkage_centroid2 = linkage(df_pca2, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid2[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid2 = fcluster(linkage_centroid2,t=thresh)
		results.append(("Agglomerative clustering Centroid 3 Components: ",normalized_mutual_info_score(classes,aggc_centroid2,average_method='geometric')))
		
		linkage_centroid3 = linkage(df_pca3, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid3[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid3 = fcluster(linkage_centroid3,t=thresh)
		results.append(("Agglomerative clustering Centroid 4 Components: ",normalized_mutual_info_score(classes,aggc_centroid1,average_method='geometric')))
		
		linkage_centroid4 = linkage(df_pca4, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid4[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid4 = fcluster(linkage_centroid4,t=thresh)
		results.append(("Agglomerative clustering Centroid 50% Components: ",normalized_mutual_info_score(classes,aggc_centroid4,average_method='geometric')))		
		
		linkage_centroid5 = linkage(df_pca5, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid5[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid5 = fcluster(linkage_centroid5,t=thresh)
		results.append(("Agglomerative clustering Centroid 60% Components: ",normalized_mutual_info_score(classes,aggc_centroid5,average_method='geometric')))		
		
		linkage_centroid6 = linkage(df_pca6, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid6[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid6 = fcluster(linkage_centroid6,t=thresh)
		results.append(("Agglomerative clustering Centroid 70% Components: ",normalized_mutual_info_score(classes,aggc_centroid6,average_method='geometric')))		
		
		linkage_centroid7 = linkage(df_pca7, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid7[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid7 = fcluster(linkage_centroid7,t=thresh)
		results.append(("Agglomerative clustering Centroid 80% Components: ",normalized_mutual_info_score(classes,aggc_centroid7,average_method='geometric')))		
		
		linkage_centroid8 = linkage(df_pca8, method='centroid') #distance of centroids metric
		dists = list(set(linkage_centroid8[:, 2]))
		thresh = (dists[1] + dists[2]) / 2
		aggc_centroid8 = fcluster(linkage_centroid8,t=thresh)
		results.append(("Agglomerative clustering Centroid 90% Components: ",normalized_mutual_info_score(classes,aggc_centroid8,average_method='geometric')))	
	
	if spec_clus_gamma05:
		#Spectral clustering Gamma 0.5
		print('#################Spectral Clustering Gamma 0.5#################')
		spec_clus3=SpectralClustering(n_clusters=3,gamma=0.5)
		spec_clus4=SpectralClustering(n_clusters=4,gamma=0.5)
		spec_clus5=SpectralClustering(n_clusters=5,gamma=0.5)
		spec_clus6=SpectralClustering(n_clusters=6,gamma=0.5)
		spec_clus7=SpectralClustering(n_clusters=7,gamma=0.5)
		
		##Spectral Clustering Gamma 0.5 2 PCA components
		#spec_clus3.fit(df_pca1)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca1)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca1)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca1)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca1)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 3 PCA components
		#spec_clus3.fit(df_pca2)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca2)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca2)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca2)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca2)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 4 PCA components
		#spec_clus3.fit(df_pca3)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca3)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca3)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca3)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca3)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 50% PCA components
		#spec_clus3.fit(df_pca4)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca4)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca4)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca4)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca4)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 60% PCA components
		#spec_clus3.fit(df_pca5)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca5)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca5)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca5)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca5)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 70% PCA components
		#spec_clus3.fit(df_pca6)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca6)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca6)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca6)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca6)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 80% PCA components
		#spec_clus3.fit(df_pca7)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca7)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca7)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca7)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca7)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 0.5 90% PCA components
		#spec_clus3.fit(df_pca8)
		#results.append(("Spectral Clustering 3 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca8)
		#results.append(("Spectral Clustering 4 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca8)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca8)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca8)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		spec_clus3.fit(df_scale)
		results.append(("Spectral Clustering 3 Clusters Gamma 0.5: ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		spec_clus4.fit(df_scale)
		results.append(("Spectral Clustering 4 Clusters Gamma 0.5: ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		spec_clus5.fit(df_scale)
		results.append(("Spectral Clustering 5 Clusters Gamma 0.5: ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		spec_clus6.fit(df_scale)
		results.append(("Spectral Clustering 6 Clusters Gamma 0.5: ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		spec_clus7.fit(df_scale)
		results.append(("Spectral Clustering 7 Clusters Gamma 0.5: ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
	
	if spec_clus_gamma10:
		#Spectral clustering Gamma 1.0
		print('#################Spectral Clustering Gamma 1.0#################')
		spec_clus3=SpectralClustering(n_clusters=3,gamma=1.0)
		spec_clus4=SpectralClustering(n_clusters=4,gamma=1.0)
		spec_clus5=SpectralClustering(n_clusters=5,gamma=1.0)
		spec_clus6=SpectralClustering(n_clusters=6,gamma=1.0)
		spec_clus7=SpectralClustering(n_clusters=7,gamma=1.0)
		
		##Spectral Clustering Gamma 1.0 2 PCA components
		#spec_clus3.fit(df_pca1)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca1)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca1)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca1)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca1)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 3 PCA components
		#spec_clus3.fit(df_pca2)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca2)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca2)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca2)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca2)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 4 PCA components
		#spec_clus3.fit(df_pca3)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca3)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca3)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca3)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca3)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 50% PCA components
		#spec_clus3.fit(df_pca4)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca4)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca4)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca4)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca4)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 60% PCA components
		#spec_clus3.fit(df_pca5)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca5)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca5)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca5)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca5)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 70% PCA components
		#spec_clus3.fit(df_pca6)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca6)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca6)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca6)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca6)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 80% PCA components
		#spec_clus3.fit(df_pca7)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca7)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca7)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca7)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca7)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.0 90% PCA components
		#spec_clus3.fit(df_pca8)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.0 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca8)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.0 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca8)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca8)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca8)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		spec_clus3.fit(df_scale)
		results.append(("Spectral Clustering 3 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		spec_clus4.fit(df_scale)
		results.append(("Spectral Clustering 4 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		spec_clus5.fit(df_scale)
		results.append(("Spectral Clustering 5 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		spec_clus6.fit(df_scale)
		results.append(("Spectral Clustering 6 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		spec_clus7.fit(df_scale)
		results.append(("Spectral Clustering 7 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
	if spec_clus_gamma15:
		#Spectral clustering Gamma 1.5
		print('#################Spectral Clustering Gamma 1.5#################')
		spec_clus3=SpectralClustering(n_clusters=3,gamma=1.5)
		spec_clus4=SpectralClustering(n_clusters=4,gamma=1.5)
		spec_clus5=SpectralClustering(n_clusters=5,gamma=1.5)
		spec_clus6=SpectralClustering(n_clusters=6,gamma=1.5)
		spec_clus7=SpectralClustering(n_clusters=7,gamma=1.5)
		
		##Spectral Clustering Gamma 1.5 2 PCA components
		#spec_clus3.fit(df_pca1)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca1)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca1)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca1)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca1)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 3 PCA components
		#spec_clus3.fit(df_pca2)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca2)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca2)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca2)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca2)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 4 PCA components
		#spec_clus3.fit(df_pca3)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca3)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca3)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca3)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca3)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 50% PCA components
		#spec_clus3.fit(df_pca4)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca4)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca4)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca4)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca4)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 60% PCA components
		#spec_clus3.fit(df_pca5)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca5)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca5)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca5)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca5)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 70% PCA components
		#spec_clus3.fit(df_pca6)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca6)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca6)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca6)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca6)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 80% PCA components
		#spec_clus3.fit(df_pca7)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca7)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca7)
		#results.append(("Spectral Clustering 5 Clusters Gamma 1.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca7)
		#results.append(("Spectral Clustering 6 Clusters Gamma 1.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca7)
		#results.append(("Spectral Clustering 7 Clusters Gamma 1.5 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 1.5 90% PCA components
		#spec_clus3.fit(df_pca8)
		#results.append(("Spectral Clustering 3 Clusters Gamma 1.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca8)
		#results.append(("Spectral Clustering 4 Clusters Gamma 1.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca8)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca8)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca8)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))	
		
		spec_clus3.fit(df_scale)
		results.append(("Spectral Clustering 3 Clusters Gamma 1.5: ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		spec_clus4.fit(df_scale)
		results.append(("Spectral Clustering 4 Clusters Gamma 1.5: ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		spec_clus5.fit(df_scale)
		results.append(("Spectral Clustering 5 Clusters Gamma 1.5: ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		spec_clus6.fit(df_scale)
		results.append(("Spectral Clustering 6 Clusters Gamma 1.5: ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		spec_clus7.fit(df_scale)
		results.append(("Spectral Clustering 7 Clusters Gamma 1.5: ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
	
	if spec_clus_gamma20:
		#Spectral clustering Gamma 2.0
		print('#################Spectral Clustering Gamma 2.0#################')
		spec_clus3=SpectralClustering(n_clusters=3,gamma=2.0)
		spec_clus4=SpectralClustering(n_clusters=4,gamma=2.0)
		spec_clus5=SpectralClustering(n_clusters=5,gamma=2.0)
		spec_clus6=SpectralClustering(n_clusters=6,gamma=2.0)
		spec_clus7=SpectralClustering(n_clusters=7,gamma=2.0)
		
		##Spectral Clustering Gamma 2.0 2 PCA components
		#spec_clus3.fit(df_pca1)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca1)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca1)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca1)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca1)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 2 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 3 PCA components
		#spec_clus3.fit(df_pca2)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca2)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca2)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca2)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca2)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 3 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 4 PCA components
		#spec_clus3.fit(df_pca3)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca3)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca3)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca3)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca3)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 4 PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 50% PCA components
		#spec_clus3.fit(df_pca4)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca4)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca4)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca4)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca4)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 50% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 60% PCA components
		#spec_clus3.fit(df_pca5)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca5)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca5)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca5)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca5)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 60% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 70% PCA components
		#spec_clus3.fit(df_pca6)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca6)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca6)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca6)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca6)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 70% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 80% PCA components
		#spec_clus3.fit(df_pca7)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca7)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca7)
		#results.append(("Spectral Clustering 5 Clusters Gamma 2.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca7)
		#results.append(("Spectral Clustering 6 Clusters Gamma 2.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca7)
		#results.append(("Spectral Clustering 7 Clusters Gamma 2.0 80% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		##Spectral Clustering Gamma 2.0 90% PCA components
		#spec_clus3.fit(df_pca8)
		#results.append(("Spectral Clustering 3 Clusters Gamma 2.0 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		#spec_clus4.fit(df_pca8)
		#results.append(("Spectral Clustering 4 Clusters Gamma 2.0 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		#spec_clus5.fit(df_pca8)
		#results.append(("Spectral Clustering 5 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		#spec_clus6.fit(df_pca8)
		#results.append(("Spectral Clustering 6 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		#spec_clus7.fit(df_pca8)
		#results.append(("Spectral Clustering 7 Clusters Gamma 0.5 90% PCA Components ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
		spec_clus3.fit(df_scale)
		results.append(("Spectral Clustering 3 Clusters Gamma 2.0: ", normalized_mutual_info_score(classes,spec_clus3.labels_,average_method='geometric')))
		spec_clus4.fit(df_scale)
		results.append(("Spectral Clustering 4 Clusters Gamma 2.0: ", normalized_mutual_info_score(classes,spec_clus4.labels_,average_method='geometric')))
		spec_clus5.fit(df_scale)
		results.append(("Spectral Clustering 5 Clusters Gamma 2.0: ", normalized_mutual_info_score(classes,spec_clus5.labels_,average_method='geometric')))
		spec_clus6.fit(df_scale)
		results.append(("Spectral Clustering 6 Clusters Gamma 2.0: ", normalized_mutual_info_score(classes,spec_clus6.labels_,average_method='geometric')))
		spec_clus7.fit(df_scale)
		results.append(("Spectral Clustering 7 Clusters Gamma 2.0: ", normalized_mutual_info_score(classes,spec_clus7.labels_,average_method='geometric')))
		
	print("Finised")
	print("Processing data")
	sorted_results=sorted(results,key=lambda item:item[1],reverse=True)
	print(sorted_results)
