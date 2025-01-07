The project implements the Bradley-Fayyad-Reina (BFR) clustering algorithm to cluster data from a synthetic 
dataset. The goal is to apply the BFR algorithm to partition the data into clusters, while handling outliers 
and ensuring efficient processing. The algorithm works by iteratively loading portions of the dataset, 
applying K-means clustering with varying numbers of clusters, and refining the clusters based on Mahalanobis 
distance. The process involves three key sets: Discard Set (DS), Compression Set (CS), and Retained Set (RS), 
with statistics such as the number of points and cluster summaries. The results, including intermediate steps 
and final cluster assignments, are output to a text file.
