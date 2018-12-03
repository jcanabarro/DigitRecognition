import sklearn
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from Classifiers import RandomForest, MultilayerPerceptron
from Extractors import PCAExtractor, CircleExtractor

images_dataset = {}


def read_image(path, folder):
    global images_dataset
    grayscale_image = misc.imread(path, flatten=True)
    images_dataset[folder].append(misc.imresize(grayscale_image, size=(50, 50)))
    return misc.imresize(grayscale_image, size=(50, 50)).flatten()


def plot_graph():
    for key, value in loaded_images.items():
        x, y = [i[0] for i in loaded_images[key]], [i[1] for i in loaded_images[key]]

        plt.scatter(x, y, label=key)
    plt.legend()
    plt.show()


def split_data(data):
    X, Y = [], []
    for key, value in data.items():
        X += [i for i in data[key]]
        Y += [key] * len(data[key])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.5, random_state=42)
    x_validation, x_test, y_validation, y_test = sklearn.model_selection.train_test_split(
        x_test, y_test, test_size=0.5, random_state=42)
    return x_train, x_validation, x_test, y_train, y_validation, y_test


if __name__ == "__main__":
    image_folder = ['zero', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
    loaded_images = {folder: [] for folder in image_folder}
    images_dataset = {folder: [] for folder in image_folder}
    all_images = []

    for folder in image_folder:
        for i in range(1, 300):
            try:
                img = read_image("./imagens/{}/{}.bmp".format(folder, i), folder)
                loaded_images[folder].append(img)
                all_images.append(img)
            except:
                pass

    ce = CircleExtractor(3, 3, 10)
    pca = PCAExtractor(all_images)

    for key, value in loaded_images.items():
        loaded_images[key] = pca.transform(loaded_images[key])
        # images_dataset[key] = ce.transform(images_dataset[key], 50, 50)
    mean_mlp, mean_rf = [], []
    proba_prod, proba_sum = [], []

    for i in range(0, 9):
        x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(loaded_images)
        # x_train, x_validation, x_test, y_train, y_validation, y_test = split_data(images_dataset)

        print("AQUI")
        rf = RandomForest(x_train, y_train)
        mlp = MultilayerPerceptron(x_train, y_train)

        print('Random Forest: ', rf.best_parameters(x_validation, y_validation))
        print('MLP:           ', mlp.best_parameters(x_validation, y_validation))

        mlp_mean = mlp.score(x_test, y_test)
        rf_mean = rf.score(x_test, y_test)

        mlp_proba = mlp.predict_proba(x_test)
        rf_proba = rf.predict_proba(x_test)

        combine_list = [mlp_proba, rf_proba]

        proba_sum.append(np.argmax(np.sum(combine_list, axis=0), axis=-1))
        proba_prod.append(np.argmax(np.prod(combine_list, axis=0), axis=-1))

        print('Random Forest:', rf_mean)
        print('MLP:          ', mlp_mean)
        mean_mlp.append(mlp_mean)
        mean_rf.append(rf_mean)

    print('rf')
    for v in mean_rf:
        print('%.5f' % v)

    print('mlp')
    for v in mean_mlp:
        print('%.5f' % v)

    print('Random Forest:', np.mean(mean_rf), np.std(mean_rf))
    print('MLP:          ', np.mean(mean_mlp), np.std(mean_mlp))
    print('Conbiner Rules')
    print('Prod:', np.mean(proba_sum), np.std(proba_sum))
    print('Sum: ', np.mean(proba_prod), np.std(proba_prod))
