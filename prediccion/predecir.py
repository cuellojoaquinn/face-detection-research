
def predict_svm(model,scaler, X):
    """
    Realiza predicciones usando el SVM entrenado.

    Parámetros:
    - model: clasificador SVM entrenado
    - scaler: objeto StandardScaler usado en entrenamiento
    - X: datos a predecir (sin normalizar)

    Retorna:
    - y_pred: predicciones del modelo
    """

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred