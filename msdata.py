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
	
	#Shuffle data set, same each run
	df=df.sample(frac=1,random_state=1)
	classes=np.array(df.pop('class'))-1 #Pop the classes
	ids=df.pop('id') #Pop the id
	cols=df.columns #Save columns
	
	#Check null values
	for column in df.columns:
		if(df[column].isnull().any()==True):
			print(column)
	
	#Check data types
	for column in df.columns:
		if(df[column].dtype!=float):
			print(column)
			
	#Seems everything ok
	#df.describe().to_csv("Description.csv")
	
	
	#Time to edit the data, scaling variables
	mmscaler = MinMaxScaler()
	sdscaler = StandardScaler()
	df_scale=pd.DataFrame(sdscaler.fit_transform(df),columns=cols)
	#print(df_scale)
	
	##Let's reduce to a 2-dimensional dataset
	#pca_2 = PCA(n_components=2)
	#df_pca = pd.DataFrame(pca_2.fit_transform(df_scale))
	#print(df_pca)
	#print(pca_2.explained_variance_ratio_)
	#df_pca.plot.scatter(0,1)
	#plt.show()
	
	#We are losing too much information
	
	#Trying with svd
	#svd_2 = TruncatedSVD(n_components=2)
	#df_svd = pd.DataFrame(svd_2.fit_transform(df_scale))
	#print(df_svd)
	#print(svd_2.explained_variance_ratio_)
	#df_svd.plot.scatter(0,1)
	#plt.show()
	
	#We got same image, I'm staying with PCA
	
	#pca=PCA(.8) #Keep 80% variance
	#print(len(pca.explained_variance_ratio_)) #Just to see the number of components
	pca=PCA(n_components=2)
	df_pca = pd.DataFrame(pca.fit_transform(df_scale))
	
	
	print('#################Kmeans#################')
	##Apply kmeans but first let's get the optimal number of clusters
	
	#silhouette_scores=[]
	#davies_bouldin_scores=[]
	#calinski_harabasz_scores=[]
	#for k in range(2, 20):
	    #kmeans = KMeans(n_clusters=k)
	    #kmeans.fit(df_pca)
	    #silhouette_scores.append(silhouette_score(df_pca, kmeans.labels_))
	    #davies_bouldin_scores.append(davies_bouldin_score(df_pca,kmeans.labels_))
	    #calinski_harabasz_scores.append(calinski_harabasz_score(df_pca,kmeans.labels_))
	
	##Plots, we want high-high-low
	#fig = plt.figure(figsize=(15, 5))
	#plt.plot(range(2, 20), silhouette_scores)
	#plt.grid(True)
	#plt.title('Get the optimal n_clusters')
	#plt.xlabel('N_clusters')
	#plt.ylabel('Silhoutte Score')
	#plt.show()
	
	#fig = plt.figure(figsize=(15, 5))
	#plt.plot(range(2, 20), calinski_harabasz_scores)
	#plt.grid(True)
	#plt.title('Get the optimal n_clusters')
	#plt.xlabel('N_clusters')
	#plt.ylabel('Calinski Harabasz Score')
	#plt.show()
	
	
	#fig = plt.figure(figsize=(15, 5))
	#plt.plot(range(2, 20), davies_bouldin_scores)
	#plt.grid(True)
	#plt.title('Get the optimal n_clusters')
	#plt.xlabel('N_clusters')
	#plt.ylabel('Davies Bouldin Score')
	#plt.show()
		
	
	#We go with 3 clusters, kmeans
	kmeans = KMeans(n_clusters=3)
	kmeans.fit(df_pca)
	print(normalized_mutual_info_score(classes,kmeans.labels_,average_method='geometric'))
	
	#Agglomerative clustering
	print('#################Agglomerative Clustering#################')
	#silhouette_scores=[]
	#davies_bouldin_scores=[]
	#calinski_harabasz_scores=[]
	#for k in range(2, 20):
	    #spec_clus = SpectralClustering(n_clusters=k)
	    #spec_clus.fit(df_pca)
	    #silhouette_scores.append(silhouette_score(df_pca, spec_clus.labels_))
	    #davies_bouldin_scores.append(davies_bouldin_score(df_pca,spec_clus.labels_))
	    #calinski_harabasz_scores.append(calinski_harabasz_score(df_pca,spec_clus.labels_))
	
	##Plots, we want high-high-low
	#fig = plt.figure(figsize=(15, 5))
	#plt.plot(range(2, 20), silhouette_scores)
	#plt.grid(True)
	#plt.title('Get the optimal n_clusters')
	#plt.xlabel('N_clusters')
	#plt.ylabel('Silhoutte Score')
	#plt.show()
	
	#fig = plt.figure(figsize=(15, 5))
	#plt.plot(range(2, 20), calinski_harabasz_scores)
	#plt.grid(True)
	#plt.title('Get the optimal n_clusters')
	#plt.xlabel('N_clusters')
	#plt.ylabel('Calinski Harabasz Score')
	#plt.show()
	
	
	#fig = plt.figure(figsize=(15, 5))
	#plt.plot(range(2, 20), davies_bouldin_scores)
	#plt.grid(True)
	#plt.title('Get the optimal n_clusters')
	#plt.xlabel('N_clusters')
	#plt.ylabel('Davies Bouldin Score')
	#plt.show()
	
	#0.85
	#aggc_single = AgglomerativeClustering(n_clusters=4,affinity='cosine',linkage='single') #single-linkage metric
	#aggc_single.fit(df_pca)
	#print(normalized_mutual_info_score(classes,aggc_single.labels_,average_method='geometric'))

	#0.7
	#aggc_complete = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete') #complete-linkage metric
	#aggc_complete.fit(df_pca)
	#print(normalized_mutual_info_score(classes,aggc_complete.labels_,average_method='geometric'))
	
	#0.79
	#aggc_avg = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='average') #average-linkage metric
	#aggc_avg.fit(df_pca)
	#print(normalized_mutual_info_score(classes,aggc_avg.labels_,average_method='geometric'))
	
	#linkage_centroid = linkage(df_pca ,method='centroid') #distance of centroids metric
	#dists = list(set(linkage_centroid[:, 2]))
	#thresh = (dists[1] + dists[2]) / 2
	#aggc_centroid = fcluster(linkage_centroid,t=thresh)
	#print(normalized_mutual_info_score(classes,aggc_centroid,average_method='geometric'))
	
	##Spectral clustering
	#print('#################Spectral Clustering#################')
	#spec_clus=SpectralClustering(n_clusters=3,gamma=1.0)
	#spec_clus.fit(df_pca)
	#print(normalized_mutual_info_score(classes,spec_clus.labels_,average_method='geometric'))
	
	#spec_clus=SpectralClustering(n_clusters=3,gamma=0.5)
	#spec_clus.fit(df_pca)
	#print(normalized_mutual_info_score(classes,spec_clus.labels_,average_method='geometric'))
	
	#spec_clus=SpectralClustering(n_clusters=3,gamma=1.5)
	#spec_clus.fit(df_pca)
	#print(normalized_mutual_info_score(classes,spec_clus.labels_,average_method='geometric'))

