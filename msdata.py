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
	ignore=True #We ignore the plots
	test=False #We ignore the tests to get the final result
	
	#Read df
	df=pd.read_csv('msdata.csv',na_values=['?','#','&','@'],keep_default_na=True)
	
	classes=np.array(df.pop('class')) #Pop the classes
	ids=df.pop('id') #Pop the id
	cols=df.columns #Save columns
	
	#Check null values
	for column in cols:
		if(df[column].isnull().any()==True):
			print(column)
	
	#Check data types
	for column in cols:
		if(df[column].dtype!=float):
			print(column)
			
	#Seems everything ok
	
	#Time to edit the data, scaling variables
	sdscaler = StandardScaler()
	df_scale=pd.DataFrame(sdscaler.fit_transform(df),columns=cols)
	
	#Let's reduce to a 2-dimensional dataset its been the most consistent of the choices
	pca_2 = PCA(n_components=2)
	df_pca = pd.DataFrame(pca_2.fit_transform(df_scale))
	
	#Plot of the dataset
	if(not ignore):
		df_pca.plot.scatter(0,1,title='Plot of the data transformed')
		plt.xlabel('First Component')
		plt.ylabel('Second Component')
		plt.show()
	
	
	print('#################Kmeans#################')
	#Plot
	if(not ignore):
		silhouette_scores=[]
		davies_bouldin_scores=[]
		calinski_harabasz_scores=[]
		for k in range(2, 20):
		    kmeans = KMeans(n_clusters=k)
		    kmeans.fit(df_pca)
		    silhouette_scores.append(silhouette_score(df_pca, kmeans.labels_))
		    davies_bouldin_scores.append(davies_bouldin_score(df_pca,kmeans.labels_))
		    calinski_harabasz_scores.append(calinski_harabasz_score(df_pca,kmeans.labels_))
		
		#Plots, we want high-high-low
		fig = plt.figure(figsize=(15, 5))
		plt.plot(range(2, 20), silhouette_scores)
		plt.grid(True)
		plt.title('Get the optimal n_clusters')
		plt.xlabel('N_clusters')
		plt.ylabel('Silhoutte Score')
		plt.show()
		
		fig = plt.figure(figsize=(15, 5))
		plt.plot(range(2, 20), calinski_harabasz_scores)
		plt.grid(True)
		plt.title('Get the optimal n_clusters')
		plt.xlabel('N_clusters')
		plt.ylabel('Calinski Harabasz Score')
		plt.show()
		
		
		fig = plt.figure(figsize=(15, 5))
		plt.plot(range(2, 20), davies_bouldin_scores)
		plt.grid(True)
		plt.title('Get the optimal n_clusters')
		plt.xlabel('N_clusters')
		plt.ylabel('Davies Bouldin Score')
		plt.show()
	
	if(not test):
		#Store the final results
		results=[]
		
		#Do the best one according to metrics
		kmeans_def=KMeans(n_clusters=5)
		kmeans_def.fit(df_pca)
		results.append(("Kmeans 5 clusters",normalized_mutual_info_score(classes,kmeans_def.labels_,average_method='geometric')))
		
	#Agglomerative clustering
	print('#################Agglomerative Clustering#################')
	#Plots
	if(not ignore):
		linkages=['single','average','complete']
		for link in linkages:
			silhouette_scores=[]
			davies_bouldin_scores=[]
			calinski_harabasz_scores=[]
			for k in range(2, 20):
			    aggc_clus = AgglomerativeClustering(n_clusters=k,affinity='cosine',linkage=link)
			    aggc_clus.fit(df_pca)
			    silhouette_scores.append(silhouette_score(df_pca, aggc_clus.labels_))
			    davies_bouldin_scores.append(davies_bouldin_score(df_pca,aggc_clus.labels_))
			    calinski_harabasz_scores.append(calinski_harabasz_score(df_pca,aggc_clus.labels_))
			
			#Plots, we want high-high-low
			fig = plt.figure(figsize=(15, 5))
			plt.plot(range(2, 20), silhouette_scores)
			plt.grid(True)
			plt.title('Get the optimal n_clusters')
			plt.xlabel('N_clusters')
			plt.ylabel('Silhoutte Score')
			plt.show()
			
			fig = plt.figure(figsize=(15, 5))
			plt.plot(range(2, 20), calinski_harabasz_scores)
			plt.grid(True)
			plt.title('Get the optimal n_clusters')
			plt.xlabel('N_clusters')
			plt.ylabel('Calinski Harabasz Score')
			plt.show()
			
			
			fig = plt.figure(figsize=(15, 5))
			plt.plot(range(2, 20), davies_bouldin_scores)
			plt.grid(True)
			plt.title('Get the optimal n_clusters')
			plt.xlabel('N_clusters')
			plt.ylabel('Davies Bouldin Score')
			plt.show()
	
	if(not test):
		#Do the best ones according to metrics
		aggc_clus_single = AgglomerativeClustering(n_clusters=6,affinity='cosine',linkage='single')
		aggc_clus_single.fit(df_pca)
		results.append(("Agglomerative clustering 6 Clusters, Single linkage, Cosine: ",normalized_mutual_info_score(classes,aggc_clus_single.labels_,average_method='geometric')))
		
		aggc_clus_average = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='average')
		aggc_clus_average.fit(df_pca)
		results.append(("Agglomerative clustering 5 Clusters, Average linkage, Cosine: ",normalized_mutual_info_score(classes,aggc_clus_average.labels_,average_method='geometric')))
		
		aggc_clus_complete = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='complete')
		aggc_clus_complete.fit(df_pca)
		results.append(("Agglomerative clustering 5 Clusters, Complete linkage, Cosine: ",normalized_mutual_info_score(classes,aggc_clus_complete.labels_,average_method='geometric')))
		
	#Spectral_Clustering
	print('#################Spectral Clustering#################')
	#Plot
	if(not ignore):
		gammas=[0.5,1.0,1.5,2.0]
		for gamma in gammas:
			silhouette_scores=[]
			davies_bouldin_scores=[]
			calinski_harabasz_scores=[]
			for k in range(2, 20):
			    spec_clus = SpectralClustering(n_clusters=k,gamma=gamma)
			    spec_clus.fit(df_pca)
			    silhouette_scores.append(silhouette_score(df_pca, spec_clus.labels_))
			    davies_bouldin_scores.append(davies_bouldin_score(df_pca,spec_clus.labels_))
			    calinski_harabasz_scores.append(calinski_harabasz_score(df_pca,spec_clus.labels_))
			
			#Plots, we want high-high-low
			fig = plt.figure(figsize=(15, 5))
			plt.plot(range(2, 20), silhouette_scores)
			plt.grid(True)
			plt.title('Get the optimal n_clusters')
			plt.xlabel('N_clusters')
			plt.ylabel('Silhoutte Score')
			plt.show()
			
			fig = plt.figure(figsize=(15, 5))
			plt.plot(range(2, 20), calinski_harabasz_scores)
			plt.grid(True)
			plt.title('Get the optimal n_clusters')
			plt.xlabel('N_clusters')
			plt.ylabel('Calinski Harabasz Score')
			plt.show()
			
			
			fig = plt.figure(figsize=(15, 5))
			plt.plot(range(2, 20), davies_bouldin_scores)
			plt.grid(True)
			plt.title('Get the optimal n_clusters')
			plt.xlabel('N_clusters')
			plt.ylabel('Davies Bouldin Score')
			plt.show()
		
	if(not test):
		#Do the best one according to metrics
		spec_clus05=SpectralClustering(n_clusters=5,gamma=0.5)
		spec_clus05.fit(df_pca)
		results.append(("Spectral Clustering 5 Clusters Gamma 0.5: ", normalized_mutual_info_score(classes,spec_clus05.labels_,average_method='geometric')))
		
		spec_clus10=SpectralClustering(n_clusters=6,gamma=1.0)
		spec_clus10.fit(df_pca)
		results.append(("Spectral Clustering 6 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,spec_clus10.labels_,average_method='geometric')))
		
		spec_clus15=SpectralClustering(n_clusters=5,gamma=1.5)
		spec_clus15.fit(df_pca)
		results.append(("Spectral Clustering 5 Clusters Gamma 1.5: ", normalized_mutual_info_score(classes,spec_clus15.labels_,average_method='geometric')))
		
		spec_clus20=SpectralClustering(n_clusters=7,gamma=2.0)
		spec_clus20.fit(df_pca)
		results.append(("Spectral Clustering 7 Clusters Gamma 2.0: ", normalized_mutual_info_score(classes,spec_clus20.labels_,average_method='geometric')))
		
		#And here are the results
		sorted_results=sorted(results,key=lambda item:item[1],reverse=True)
		print(sorted_results)
		
	
	#Create color points
	color = {'1':'red', '2':'green', '3':'blue','4':'yellow','5':'magenta','6':'cyan'}
	colors=[]
	for elem in classes:
		colors.append(color[str(elem)])
		
	df_pca.plot.scatter(0,1,title='Plot of the data transformed with labels',c=colors)
	plt.xlabel('First Component')
	plt.ylabel('Second Component')
	plt.show()
	
	#Seems the best one is spectral clustering, 6 clusters gamma=1.0
	spec_clus10=SpectralClustering(n_clusters=6,gamma=1.0)
	spec_clus10.fit(df_pca)
	labels=spec_clus10.labels_
	labels+=1 #This is for the key of the colors dictionary
	
	colors=[]
	for elem in labels:
		colors.append(color[str(elem)])
	
	df_pca.plot.scatter(0,1,title='Plot of the prediction',c=colors)
	plt.xlabel('First Component')
	plt.ylabel('Second Component')
	plt.show()
	
	print("Spectral Clustering 6 Clusters Gamma 1.0: ", normalized_mutual_info_score(classes,labels,average_method='geometric'))
	
	#Save data in txt file
	f = open("msdata.txt",'w')
	for elem in labels:
		f.write(str(elem)+'\n')
	f.close()
	
