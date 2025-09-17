def region_sum(integral, x1, y1, x2, y2):
    """Calcula la suma de intensidades en un rectángulo, ajustando los límites automáticamente."""
    # Limitar los índices a los bordes de la imagen
    y_max, x_max = integral.shape
    x1 = max(0, min(x1, x_max-1))
    x2 = max(0, min(x2, x_max-1))
    y1 = max(0, min(y1, y_max-1))
    y2 = max(0, min(y2, y_max-1))

    total = integral[y2, x2]
    if x1 > 0:
        total -= integral[y2, x1-1]
    if y1 > 0:
        total -= integral[y1-1, x2]
    if x1 > 0 and y1 > 0:
        total += integral[y1-1, x1-1]
    return total

class HaarFeature:
    def __init__(self, ftype, i, j, w, h, value):
        self.type = ftype
        self.i = i
        self.j = j
        self.w = w
        self.h = h
        self.value = value

    def __repr__(self):
        return f"Type={self.type} Pos=({self.i},{self.j}) w={self.w} h={self.h} Value={self.value}"


def compute_features(ii):
    n_y, n_x = ii.shape
    features = []

    # Tipo (a): 2 rectángulos horizontales
    for i in range(n_y):
        for j in range(n_x):
            for w in range(1, n_x - j + 1):
                if j + 2 * w > n_x:
                    break
                for h in range(1, n_y - i + 1):
                    S1 = region_sum(ii, j, i, j + w - 1, i + h - 1)
                    S2 = region_sum(ii, j + w, i, j + 2*w - 1, i + h - 1)
                    value = S1 - S2
                    features.append(HaarFeature(1, i, j, w, h, value))

    # Tipo (b): 3 rectángulos horizontales
    for i in range(n_y):
        for j in range(n_x):
            for w in range(1, n_x - j + 1):
                if j + 3 * w > n_x:
                    break
                for h in range(1, n_y - i + 1):
                    S1 = region_sum(ii, j, i, j + w - 1, i + h - 1)
                    S2 = region_sum(ii, j + w, i, j + 2*w - 1, i + h - 1)
                    S3 = region_sum(ii, j + 2*w, i, j + 3*w - 1, i + h - 1)
                    value = S1 - S2 + S3
                    features.append(HaarFeature(2, i, j, w, h, value))


    # Tipo (c): 2 rectángulos verticales
    for i in range(n_y):
        for j in range(n_x):
            for h in range(1, n_y - i + 1):
                if i + 2 * h > n_y:
                    break
                for w in range(1, n_x - j + 1):
                    S1 = region_sum(ii, j, i, j + w - 1, i + h - 1)
                    S2 = region_sum(ii, j, i + h, j + w - 1, i + 2*h - 1)
                    value = S1 - S2
                    features.append(HaarFeature(3, i, j, w, h, value))

    # Tipo (d): 3 rectángulos verticales
    for i in range(n_y):
        for j in range(n_x):
            for h in range(1, n_y - i + 1):
                if i + 3 * h > n_y:
                    break
                for w in range(1, n_x - j + 1):
                    S1 = region_sum(ii, j, i, j + w - 1, i + h - 1)
                    S2 = region_sum(ii, j, i + h, j + w - 1, i + 2*h - 1)
                    S3 = region_sum(ii, j, i + 2*h, j + w - 1, i + 3*h - 1)
                    value = S1 - S2 + S3
                    features.append(HaarFeature(4, i, j, w, h, value))

    # Tipo (e): 4 rectángulos en cuadrícula 2x2
    for i in range(n_y):
        for j in range(n_x):
            for h in range(1, n_y - i + 1):
                if i + 2 * h > n_y:
                    break
                for w in range(1, n_x - j + 1):
                    if j + 2 * w > n_x:
                        break
                    S1 = region_sum(ii, j, i, j + w - 1, i + h - 1)
                    S2 = region_sum(ii, j, i + h, j + w - 1, i + 2*h - 1)
                    S3 = region_sum(ii, j + w, i, j + 2*w - 1, i + h - 1)
                    S4 = region_sum(ii, j + w, i + h, j + 2*w - 1, i + 2*h - 1)
                    value = S1 - S2 - S3 + S4
                    features.append(HaarFeature(5, i, j, w, h, value))

    return features

# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    import numpy as np


    # Creamos una imagen de ejemplo pequeña (4x4) para probar
    img = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 1, 1, 1],
        [2, 2, 2, 2]
    ], dtype=np.float64)

    # Calculamos la imagen integral
    ii = img.cumsum(axis=0).cumsum(axis=1)

    # Calculamos Haar features
    features = compute_features(ii)

    # Mostrar resultados
    print(f"Total features generadas: {len(features)}")
    print(" features:")
    for f in features:
        print(f)
