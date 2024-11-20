import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_data(random_state=2024):

    np.random.seed(random_state)

    # Generate three clusters with distinct means and covariance matrices
    mean1 = [10, 10, 10]
    cov1 = [[2, 1.6, 1], [1.6, 2, 0.6], [1, 0.6, 2]]
    data1 = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [20, 20, 10]
    cov2 = [[2, -1, 0], [-1, 2, 0], [0, 0, 2]]
    data2 = np.random.multivariate_normal(mean2, cov2, 100)

    mean3 = [15, 15, 25]
    cov3 = [[2, 0, 0], [0, 2, 0], [0, 0, 10]]
    data3 = np.random.multivariate_normal(mean3, cov3, 100)

    # Combine the data and assign labels to each cluster
    data = np.vstack((data1, data2, data3))
    labels = np.array([0]*100 + [1]*100 + [2]*100)  # 0 for cluster 1, 1 for cluster 2, 2 for cluster 3
    return data, labels


# Function to standardize the dataset
def standardize_data(data):

    npdata = np.array(data)
    stdScaler = StandardScaler()
    data_scaled = stdScaler.fit_transform(npdata)
    return data_scaled

    return data_scaled

def apply_pca(data_scaled, n_components):

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)


    return data_pca, pca


# Function to plot the original 3D data
def plot_original_data(data, labels):

    three_dfig=plt.figure(figsize=(6,6))
    ax = three_dfig.add_subplot(projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='plasma', marker='x')

    colorlabel = plt.colorbar(scatter, ax=ax, fraction=0.029, pad=0.12)
    colorlabel.set_label('Cluster Colors Label')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title("Plot of 3D Clusters dataset ")

    min_x=np.min(data[:,0])
    min_y=np.min(data[:,1])
    min_z=np.min(data[:,2])
    max_x=np.max(data[:,0])
    max_y=np.max(data[:,1])
    max_z=np.max(data[:,2])


    ax.set_xlim([min_x,max_x])
    ax.set_ylim([min_y,max_y])
    ax.set_zlim([min_z,max_z])

    plt.show()


#XY projection means only feature 1 and feature 2 will be accounted for.like looking from above.

def plot_xy_projection(data, labels):

    xy_data = np.column_stack((data[:, 0], data[:, 1]))

    plt.scatter(xy_data[:, 0], xy_data[:, 1], c=labels, cmap='plasma')
    plt.xlabel('Feature 1') 
    plt.ylabel('Feature 2')  
    plt.title('XY Looking From Above Projection of Dataset ') 
    plt.colorbar(label='Cluster Colors Label') 

    min_x=np.min(xy_data[:,0])
    min_y=np.min(xy_data[:,1])
    max_x=np.max(xy_data[:,0])
    max_y=np.max(xy_data[:,1])

    plt.xlim([min_x,max_x])
    plt.ylim([min_y,max_y])

    plt.show()  # Display the plot

# Function to plot the XZ projection
def plot_xz_projection(data, labels):

    xz_data = np.column_stack((data[:, 0], data[:, 2]))
    
    plt.scatter(xz_data[:, 0], xz_data[:, 1], c=labels, cmap='plasma')
    plt.xlabel('Feature 1') 
    plt.ylabel('Feature 3')  
    plt.title('XZ Looking From Front Projection of Dataset ') 
    plt.colorbar(label='Cluster Colors Label') 

    min_x=np.min(xz_data[:,0])
    min_z=np.min(xz_data[:,1])
    max_x=np.max(xz_data[:,0])
    max_z=np.max(xz_data[:,1])

    plt.xlim([min_x,max_x])
    plt.ylim([min_z,max_z])

    plt.show()  # Display the plot


# Function to plot the YZ projection
def plot_yz_projection(data, labels):

    yz_data = np.column_stack((data[:, 1], data[:, 2]))
    
    plt.scatter(yz_data[:, 0], yz_data[:, 1], c=labels, cmap='plasma')
    plt.xlabel('Feature 2') 
    plt.ylabel('Feature 3')  
    plt.title('YZ Looking From the Side Projection of Dataset ') 
    plt.colorbar(label='Cluster Colors Label') 

    min_y=np.min(yz_data[:,0])
    min_z=np.min(yz_data[:,1])
    max_y=np.max(yz_data[:,0])
    max_z=np.max(yz_data[:,1])

    plt.xlim([min_y,max_y])
    plt.ylim([min_z,max_z])

    plt.show()  # Display the plot


def plot_pca_results(data_pca, labels):

    plt.figure(figsize=(6,6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels,cmap='plasma', marker='x')
    plt.xlabel('PC Feature 1')
    plt.ylabel('PC Feature 2')

    min_x=np.min(data_pca[:,0])
    min_y=np.min(data_pca[:,1])
    max_x=np.max(data_pca[:,0])
    max_y=np.max(data_pca[:,1])

    plt.xlim([min_x,max_x])
    plt.ylim([min_y,max_y])

    plt.show()

if __name__ == "__main__":
    np.random.seed(2024)
    data, labels = generate_data() # Generate the data and labels
    
    # Plot original 3D data and projections
    plot_original_data(data, labels)
    plot_xy_projection(data, labels)
    plot_xz_projection(data, labels)
    plot_yz_projection(data, labels)

    # Standardize the dataset
    data_scaled = standardize_data(data)

    # TO DO: Fill in appropriate value for n_components
    data_pca, pca = apply_pca(data_scaled, n_components=2)  # Apply PCA to the dataset
    
    plot_pca_results(data_pca, labels)   # Plot PCA results