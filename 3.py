import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split

df = pd.read_csv('BRCA_pam50.tsv', sep='\t', index_col=0)
X = df.iloc[:, :-1].to_numpy()
y = df['Subtype'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)

rfc = RandomForestClassifier(
    random_state=17
)
param_grid = {
    'max_depth': list(range(1, 20))
}
model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=RepeatedStratifiedKFold(n_repeats=10))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_BA = accuracy_score(y_test, y_pred)
print(test_BA)
print(model.best_params_)

