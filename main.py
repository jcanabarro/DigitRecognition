from scipy import misc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def read_image(path):
    grayscale_image = misc.imread(path, flatten=True)
    return misc.imresize(grayscale_image, size=(50, 50)).flatten()


class PCAExtractor:

    def __init__(self, X, components=2):
        self.X = X
        self.pca = PCA(n_components=components)
        self.pca.fit(self.X)

    def transform(self, X):
        return self.pca.transform(X)

if __name__ == "__main__":
    image_folder = ['zero', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
    loaded_images = {folder: [] for folder in image_folder}
    all_images = []

    for folder in image_folder:
        for i in range(1, 300):
            try:
                img = read_image("./imagens/{}/{}.bmp".format(folder, i))
                loaded_images[folder].append(img)
                all_images.append(img)
            except:
                pass

    pca = PCAExtractor(all_images)

    for key, value in loaded_images.items():
        loaded_images[key] = pca.transform(loaded_images[key])

    for key, value in loaded_images.items():
        x, y = [i[0] for i in loaded_images[key]], [i[1] for i in loaded_images[key]]

        plt.scatter(x, y, label=key)

    plt.legend()
    plt.show()
