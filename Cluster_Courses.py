import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score


# Upload Course Stats and Scale Data
df = pd.read_excel("New_CourseStats_EX.xlsx")
df.dropna(inplace=True)
course_names = df['course'].tolist()
print(course_names)
drop_columns = ['course']
df.drop(columns=drop_columns, inplace=True)
scaler = StandardScaler()
cluster_df = df
columns = cluster_df.columns.tolist()




cluster_df_scaled = scaler.fit_transform(cluster_df)

# Perform PCA 
pca = PCA(n_components=2)
principal_components = pca.fit_transform(cluster_df_scaled)
# Step 3: Extract the loadings matrix
loadings = pca.components_

#Step 4: Create a DataFrame to hold the loadings with feature names
loadings_df = pd.DataFrame(loadings.T, columns=['PC1', 'PC2'], index=df.columns)

#Step 5: Get the absolute values and sort to see top contributing variables
loadings_df['PC1_abs'] = loadings_df['PC1'].abs()
loadings_df['PC2_abs'] = loadings_df['PC2'].abs()




#Sort by the absolute contribution for each component
top_contributors_pc1 = loadings_df.sort_values(by='PC1_abs', ascending=False)['PC1']
top_contributors_pc2 = loadings_df.sort_values(by='PC2_abs', ascending=False)['PC2']

print("Top contributing variables for PC1:")
print(top_contributors_pc1)

print("\nTop contributing variables for PC2:")
print(top_contributors_pc2) 

  
# Function to find best value of k for iterations of k = n
def optimise_k_means(data, max_k):
    means = []
    inertias = []
    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        means.append(k)
        inertias.append(kmeans.inertia_)

    #Generate the elbow plot
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Cluster Elbow Plot')
    plt.grid(True)
    plt.show()

# Function to visualize cluster groups
def visualize_clusters(data, clusters, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    
    plt.show()
    


pc1 = principal_components[:, 0].reshape(-1, 1)  # Reshape to a 2D array for KMeans
optimise_k_means(principal_components, 10)

kmeans = KMeans(n_clusters=2, random_state=20)
kmeans.fit(principle_componenets)
cluster_df['kmeans_2'] = kmeans.labels_
print(cluster_df)
print(type(kmeans.labels_))

visualize_clusters(principal_components, kmeans.labels_, 'KMeans Clustering with 2 clusters')


course_cluster_df = pd.DataFrame({
    'Course_name': course_names,
    'cluster': kmeans.labels_
})

# Display the DataFrame
print(course_cluster_df)


# Unhash this if you want to save the df to excel 
#course_cluster_df.to_excel('PCA_Course_Cluster.xlsx')




# Calculate a silhouette score to determine cluster distinction
score = silhouette_score(principal_components, kmeans.labels_)
print(f"Silhouette Score: {score}")
