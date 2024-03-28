import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1.Przygotowanie danych
df = pd.read_csv('Iris.csv')
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Liczba próbek dla każdej klasy:")    # Sprawdzenie zbalansowania danych
print(y.value_counts())

# 2.Eksploracja danych
print("Informacje o danych:")
print(df.info())

sns.pairplot(df, hue='Species')     # Wykresy punktowe
plt.show()

# 3.Wybór modelu
model = KNeighborsClassifier(n_neighbors=3)

# 4.Trenowanie modelu
model.fit(X_train, y_train)

# 5.Ocena modelu
y_pred = model.predict(X_test)
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))
print("Macierz konfuzji:")
print(confusion_matrix(y_test, y_pred))

# 6.Interpretacja wyników
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Wyświetlanie ważności cech
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
print("Ważność cech:")
for f in range(X.shape[1]):
    print(f"{features[indices[f]]}: {importances[indices[f]]}")

# 7.Walidacja krzyżowa
scores = cross_val_score(model, X, y, cv=5)
print("Wyniki walidacji krzyżowej:", scores)
print("Średnia dokładność:", np.mean(scores))

# 8.Optymalizacja modelu
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Najlepsze parametry:", grid_search.best_params_)