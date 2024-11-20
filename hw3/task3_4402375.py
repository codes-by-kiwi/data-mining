from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Function to generate the dataset
def generate_swiss_roll(n_samples, noise=0.1, random_state=2024):
    X, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, color


# Function to apply PCA to the dataset
def apply_pca(X, n_components, random_state=2024):
    
    pipeline = make_pipeline(StandardScaler(),PCA(n_components=n_components, random_state=random_state))
    X_pca=pipeline.fit_transform(X)

    return X_pca

# Function to plot the original 3D data
def plot_3d_data(X, color):

    three_dfig=plt.figure(figsize=(6,6))
    ax = three_dfig.add_subplot(projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='plasma', marker='x')

    colorlabel = plt.colorbar(scatter, ax=ax, fraction=0.029, pad=0.12)
    colorlabel.set_label('Cluster Colors Label')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title("Plot of 3D Swiss Roll dataset ")

    min_x=np.min(X[:,0])
    min_y=np.min(X[:,1])
    min_z=np.min(X[:,2])
    max_x=np.max(X[:,0])
    max_y=np.max(X[:,1])
    max_z=np.max(X[:,2])


    ax.set_xlim([min_x,max_x])
    ax.set_ylim([min_y,max_y])
    ax.set_zlim([min_z,max_z])

    plt.show()


# Function to plot the XZ projection
def plot_xz_projection(X, color):

    xz_data = np.column_stack((X[:, 0], X[:, 2]))

    plt.scatter(xz_data[:, 0], xz_data[:, 1], c=color, cmap='plasma')
    plt.xlabel('Feature 1') 
    plt.ylabel('Feature 3')  
    plt.title('XZ Looking From Front Projection of Swiss Dataset ') 
    plt.colorbar(label='Cluster Colors Label') 

    min_x=np.min(xz_data[:,0])
    min_z=np.min(xz_data[:,1])
    max_x=np.max(xz_data[:,0])
    max_z=np.max(xz_data[:,1])

    plt.xlim([min_x,max_x])
    plt.ylim([min_z,max_z])

    plt.show()  # Display the plot 



# Function to plot the 2D PCA projection
def plot_pca_projection(X_pca, color):

    # TO DO: Use scatter plot to visualize the 2D projection from PCA
    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=color,cmap='plasma',s=10)
    plt.xlabel('PC Feature 1')
    plt.ylabel('PC Feature 2')

    min_x=np.min(X_pca[:,0])
    min_y=np.min(X_pca[:,1])
    max_x=np.max(X_pca[:,0])
    max_y=np.max(X_pca[:,1])

    plt.xlim([min_x,max_x])
    plt.ylim([min_y,max_y])

    plt.show()


if __name__ == "__main__":
    np.random.seed(2024)
    X, color = generate_swiss_roll(n_samples=1500, noise=0.1, random_state=2024)

    #TO DO: Fill in appropriate value for n_components
    X_pca = apply_pca(X, n_components=2, random_state=2024) # Apply PCA

    plot_3d_data(X, color)              # Visualize the original 3D dataset
    plot_xz_projection(X, color)        # Visualize the XZ projection
    plot_pca_projection(X_pca, color)   # Visualize the PCA 2D projection