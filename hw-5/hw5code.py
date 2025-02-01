import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать средне е двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ

    sorted_indices = np.argsort(feature_vector) # сортируем
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    if len(np.unique(feature_vector)) == 1:
        return [], [], -np.inf, -np.inf

    thresholds = (sorted_features[:-1] + sorted_features[1:]) / 2 # пороги средние между соседними значениями
    delete = sorted_features[:-1] != sorted_features[1:] # смотрим чтобы не было одинаковых соседних значений, чтобы избежать пустых поддеревьев
    thresholds = thresholds[delete] 

    R = len(target_vector)
    l_1 = np.cumsum(sorted_targets)[:-1] # кол-во 1 слева для каждого порога 
    l_0 = np.cumsum(1 - sorted_targets)[:-1] # кол-во 0 слева для каждого порога
    r_1 = np.sum(target_vector) - l_1
    r_0 = R - np.sum(target_vector) - l_0
    l_size = l_1 + l_0 # вектор размеров левых поддеревьев для каждого порога
    r_size = r_1 + r_0

    l_size[l_size == 0] = 1 # избегаю деления на 0
    r_size[r_size == 0] = 1    

    H_l = 1 - (l_1 / l_size) ** 2 - (l_0 / l_size) ** 2 
    H_r = 1 - (r_1 / r_size) ** 2 - (r_0 / r_size) ** 2
    ginis = -(l_size / R) * H_l - (r_size / R) * H_r # критерий Джини

    ginis = ginis[delete] # убираю 

    best_idx = np.argmax(ginis) # индекс лучшего порога
    gini_best = ginis[best_idx]
    threshold_best = thresholds[best_idx]

    return thresholds, ginis, threshold_best, gini_best



class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]): # 
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]): #
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0 
                    ratio[key] = current_click / current_count #
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))) #
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature]))) #
            else:
                raise ValueError

            if np.all(feature_vector == feature_vector[0]): #
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical": #
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        
        if self._min_samples_leaf is not None and (len(sub_X[split]) < self._min_samples_leaf or len(sub_X[np.logical_not(split)]) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1) #


    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_split = node["feature_split"]
        feature_value = x[feature_split]

        if self._feature_types[feature_split] == "categorical":
            if feature_value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)



class LinearRegressionTree():
    def __init__(self, feature_types, base_model_type=LinearRegression(), max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._base_model_type = base_model_type


    def _mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


    def _find_best_split(self, feature_vector, target_vector, X):
        sorted_indices = np.argsort(feature_vector)
        sorted_features = feature_vector[sorted_indices]

        quantiles = np.linspace(0.1, 0.9, 9)  # квантили от 10% до 90% с шагом 10%
        thresholds = np.quantile(sorted_features, quantiles)
        thresholds = np.unique(thresholds)

        best_mse = np.inf
        best_split, best_threshold = None, None

        for threshold in thresholds:
            split = feature_vector < threshold
            l_X = X[split]
            l_y = target_vector[split]
            r_X = X[np.logical_not(split)]
            r_y = target_vector[np.logical_not(split)]

            if len(l_y) == 0 or len(r_y) == 0:
                    continue
            
            l_model = LinearRegression().fit(l_X, l_y)
            r_model = LinearRegression().fit(r_X, r_y)

            l_mse = self._mse(l_y, l_model.predict(l_X)) 
            r_mse = self._mse(r_y, r_model.predict(r_X)) 
            mse = l_mse * (len(l_y) / len(target_vector)) + r_mse * (len(r_y) / len(target_vector))

            if mse < best_mse:
                best_mse = mse
                best_split = split
                best_threshold = threshold

        return best_mse, best_split, best_threshold


    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        feature_best, best_split, best_threshold = None, None, None
        best_mse = np.inf
        for feature in range(sub_X.shape[1]):
            feature_vector = sub_X[:, feature]
            mse, split, threshold = self._find_best_split(feature_vector, sub_y, sub_X)

            if np.all(feature_vector == feature_vector[0]): #len(np.unique(feature_vector)) == 1:
                continue

            if mse < best_mse:
                feature_best = feature
                best_split = split
                best_threshold = threshold
                best_mse = mse

        if feature_best is None:
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = best_threshold

        if self._min_samples_leaf is not None and (np.sum(best_split) < self._min_samples_leaf or np.sum(np.logical_not(best_split)) < self._min_samples_leaf):
            node["type"] = "terminal"
            node["model"] = LinearRegression().fit(sub_X, sub_y)
            return

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[best_split], sub_y[best_split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(best_split)], sub_y[np.logical_not(best_split)], node["right_child"], depth + 1)


    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict([x])[0]

        feature_split = node["feature_split"]
        feature_value = x[feature_split]

        if feature_value < node["threshold"]:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])
        

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)


    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)