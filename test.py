import numpy as np
from k_means_constrained import KMeansConstrained

# Sample zip codes, latitudes, longitudes, and demands
zip_codes = np.array(["10001", "10002", "10003", "10004", "10005", "10006", "10007", "10008"])
latitudes = np.array([40.7128, 40.7075, 40.7839, 40.7589, 40.6782, 40.7128, 40.7786, 40.7850])
longitudes = np.array([-74.0060, -73.9987, -73.9653, -73.9857, -73.9857, -74.0060, -73.9857, -73.9688])
demands = np.array([100, 200, 300, 150, 250, 100, 400, 50])

# Combine data into a single array
data = np.column_stack((latitudes, longitudes, demands))

# Print the data
print("Zip Codes:", zip_codes)
print("Latitude:", latitudes)
print("Longitude:", longitudes)
print("Demand:", demands)
print("Combined Data:", data)


# Set the number of clusters and optional constraints
n_clusters = 4
size_min = 150  # Minimum number of zip codes per cluster (optional)
size_max = 300  # Maximum number of zip codes per cluster (optional)

# Create the KMeansConstrained object
kmeans = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_max)

# Fit the model to the data (training)
kmeans.fit(data)

# Print the cluster centers (locations)
print("Cluster Centers:", kmeans.cluster_centers_)

# New data point (example) for prediction
new_data = np.array([[40.74, -74.01, 120]])  # Latitude, Longitude, Demand

# Predict the cluster label for the new data point
predicted_label = kmeans.predict(new_data)

print("Predicted Label for New Data:", predicted_label[0])


