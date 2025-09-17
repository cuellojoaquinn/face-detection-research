import numpy as np
from sklearn.model_selection import train_test_split
from svm import SVM

def main():

    #1. Dividir en conjunto de datos de entrenamiento y prueba

    #2. Enviar X_train : features e y_train: etiqueta (rostro = 1 , no rostro 0)
    clf = SVM()
    #3. Entrenamiento a partir de los datos de prueba
    clf.fit(X_train, y_train)
    #4. Predecir conjunto de datos de prueba
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    print("SVM classification accuracy", accuracy(y_test, predictions))


if __name__ == "__main__":
    main()
