import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

seed = 3
np.random.seed(seed)

# x_path = 'ENSG00000228314.1.training_geno.dat'
x_fname = sys.argv[1]
y_fname = x_fname[:-8] + 'pheno.dat'
x_test_fname = x_fname[:-17] + 'testing_geno.dat'
y_pred_fname = x_fname[:-17] + 'predicted_pheno.dat'
x_path = 'training_geno/' + x_fname
y_path = 'training_pheno/' + y_fname
x_test_path = 'testing_geno/' + x_test_fname
y_pred_path = 'predicted_pheno/' + y_pred_fname

df_x = pd.read_table(x_path, delim_whitespace=True, header=0)
X = df_x.values
df_x_test = pd.read_table(x_test_path, delim_whitespace=True, header=0)
X_test = df_x_test.values
n, p = X.shape
df_y = pd.read_table(y_path, delim_whitespace=True, header=None)
y = df_y.values[:,0]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train = X
y_train = y
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
# U, d, Vt = np.linalg.svd(X_train, full_matrices = False)
# V = Vt.transpose()
# n, p = U.shape
# U_test = np.dot(X_test, V)

# pyplot.scatter(U[:,0], y_train)
# pyplot.show()


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(5000, input_dim=p, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(3000, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(900, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(700, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(90, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(70, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
#    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))
    model.add(Dense(500, kernel_initializer='normal', activation='relu'))
    model.add(Dense(250, kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(25, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    return model


# fix random seed for reproducibility
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)

model = baseline_model()
history = model.fit(X_train, y_train, epochs=100, batch_size=len(X), verbose=2)
# pyplot.plot(history.history['mean_squared_error'])
# pyplot.plot(history.history['mean_absolute_error'])
# pyplot.show()

y_pred = model.predict(X_test)
y_pred = y_pred[:,0]
# y_pred_naive = [np.mean(y_train)] * len(y_pred)
# print("baseline mse:")
# print(mean_squared_error(y_test, y_pred_naive))
# print("nn mse:")
# print(mean_squared_error(y_test, y_pred))
# compare = np.concatenate(([y_test], [y_pred]), axis = 0)
# compare = compare.transpose()
np.savetxt(y_pred_path, y_pred)
# pyplot.scatter(y_test, prediction)
# pyplot.show()

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# # evaluate model with standardized dataset
# np.random.seed(seed)
# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
