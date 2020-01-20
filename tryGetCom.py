# Import the necessary modules and libraries
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import visualize
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import random_projection
import random
# generate data
import math
n_sample = 1000000
size_ = int(n_sample / 4)
sigma = [[0.25, 0], [0, 0.25]]
mean = [[1, 0], [0, 1], [-1, 0], [0, -1]]
X = np.zeros((n_sample, 2))
y = np.zeros((n_sample,), dtype=int)
for i in range(4):
    X[(size_ * i):(size_ * (i + 1))] = np.random.multivariate_normal(mean[i], sigma, size=size_)
    y[(size_ * i):(size_ * (i + 1))] = i

#add projection as features
proj_num = 50
sigma = [[1, 0], [0, 1]]


pro = np.zeros((n_sample, 1))
for i in range(proj_num):
    a, b = random.uniform(-1, 1), random.uniform(-1, 1)
    a, b = a/math.sqrt(a**2 + b**2), b / math.sqrt(a**2 + b**2)
    pro = np.array([X[:, 0] * a + X[:, 1] * b]).T
    X = np.hstack((X, pro))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#
# # tune max_depth = 8
# def cv_score(d):
#     clf = DecisionTreeClassifier(max_depth=d)
#     clf.fit(X_train, y_train)
#     return clf.score(X_train, y_train), clf.score(X_test, y_test)
#
#
# test_time = 10
# best_depth = np.zeros((test_time, 1))
#
# for i in range(test_time):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     depths = np.arange(5, 15)
#     scores = [cv_score(d) for d in depths]
#     tr_scores = [s[0] for s in scores]
#     te_scores = [s[1] for s in scores]
#     # find the best depth
#     tr_best_index = np.argmax(tr_scores)
#     te_best_index = np.argmax(te_scores)
#     best_depth[te_best_index] = best_depth[te_best_index] + 1
#
# best_max_depth = np.argmax(best_depth) + 5
# print("bestdepth:", best_max_depth, " bestdepth_score:", te_scores[te_best_index])
#
# depths = np.arange(5, 15)
# plt.figure(figsize=(6, 4), dpi=120)
# plt.grid()
# plt.xlabel('max depth of decison tree')
# plt.ylabel('Scores')
# plt.plot(depths, te_scores, label='test_scores')
# plt.plot(depths, tr_scores, label='train_scores')
# plt.legend()
# plt.show()
#
# # tune impurity
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#
# def minsplit_score(val):
#     clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
#     clf.fit(X_train, y_train)
#     return clf.score(X_train, y_train), clf.score(X_test, y_test)
#
# vals = np.linspace(0, 0.2, 100)
# scores = [minsplit_score(v) for v in vals]
# tr_scores = [s[0] for s in scores]
# te_scores = [s[1] for s in scores]
#
# bestmin_index = np.argmax(te_scores)
# bestscore = te_scores[bestmin_index]
#
# print("best_min_impurity:", vals[bestmin_index], "bestscore:", bestscore)
#
# plt.figure(figsize=(6, 4), dpi=120)
# plt.grid()
# plt.xlabel("min_impurity_decrease")
# plt.ylabel("Scores")
# plt.plot(vals, te_scores, label='test_scores')
# plt.plot(vals, tr_scores, label='train_scores')
# plt.legend()
# plt.show()
#
#

#
#
# from sklearn.model_selection import GridSearchCV
# SEED = 1
# dt = DecisionTreeClassifier(random_state=SEED)
#
# # define the grid of hyperparameters 'params_dt'
# params_dt = {
#             'max_depth': np.arange(8, 15),
#             'max_features': [0.13, 0.14, 0.15, 0.16, 0.17]
# }
#
# grid_dt = GridSearchCV(estimator=dt,
#                        param_grid=params_dt,
#                        scoring='accuracy',
#                        cv=5,
#                        n_jobs=-1)
# grid_dt.fit(X_train, y_train)
# best_hyperparams = grid_dt.best_params_
# print('Best hyerparameters:\n', best_hyperparams)
# best_CV_score = grid_dt.best_score_
# print('Best CV accuracy'.format(best_CV_score))
#
# best_model = grid_dt.best_estimator_
# test_acc = best_model.score(X_test,y_test)
# print("Test set accuracy of best model: {:.3f}".format(test_acc))
#



#
#
#
# clf = DecisionTreeClassifier(max_depth=8)
# # clf = DecisionTreeClassifier(max_depth=12, max_features=0.15, random_state=1)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# X_test_com = X_test[:, 0] + 1j * X_test[:, 1]
# visualize.visualize_constellation(data=list(X_test_com), labels=list(y_pred), title_string='max_depth=8')

# visualize.gen_demod_grid(points_per_dim=10, min_val=-3.5, max_val=3.5)

# with open("tree.dot", 'w') as f:
#     f = tree.export_graphviz(regr_2, out_file=f)






clf = DecisionTreeClassifier(max_depth=4)
# clf = DecisionTreeClassifier(max_depth=12, max_features=0.15, random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
X_test_com = X_test[:, 0] + 1j * X_test[:, 1]
visualize.visualize_constellation(data=list(X_test_com), labels=list(y_pred), title_string='max_depth=4')
