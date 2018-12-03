from sklearn.decomposition import PCA


class PCAExtractor:

    def __init__(self, X, components=2):
        self.X = X
        self.pca = PCA(n_components=components, svd_solver='full')
        self.pca.fit(self.X)

    def transform(self, X):
        return self.pca.transform(X)


class CircleExtractor:

    def __init__(self, nb_circle_h, nb_circle_v, radius):
        self.nb_circle_h = nb_circle_h
        self.nb_circle_v = nb_circle_v
        self.radius = radius

    def transform(self, X, nb_lines, nb_cols):
        features_list = []
        for x in X:
            features = []
            for i in range(nb_cols // self.nb_circle_h, nb_cols, nb_cols // self.nb_circle_h):
                for j in range(nb_lines // self.nb_circle_v, nb_lines, nb_lines // self.nb_circle_v):
                    feature = 0
                    for index_l, line in enumerate(x):
                        for index_c, pixel in enumerate(line):
                            if (index_l - i) ** 2 + (index_c - j) ** 2 < self.radius:
                                feature += pixel
                    features.append(feature)
            features_list.append(features)
        return features_list
