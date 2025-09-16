from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

def training_svm(haarFeatures, labels):
    """
    Entrena un clasificador SVM usando Haar features.
    Retorna el modelo, el scaler y los datos de prueba.
    """

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        haarFeatures, labels, test_size=0.2, random_state=42
    )

    # Normalizar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Entrenar SVM
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    return model, scaler, X_test, y_test
