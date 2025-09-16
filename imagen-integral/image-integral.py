from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def integral_image(img):
    return img.astype(np.float64).cumsum(axis=0).cumsum(axis=1)

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

# Cargar imagen en escala de grises con Pillow
image_path = r"C:\Users\MáximoPerea\Documents\GitHub\face-detection-research\imagen-integral\imagenPrueba2.jpg"
img = Image.open(image_path).convert("L")
img = np.array(img, dtype=np.float64)

# Calcular imagen integral
I = integral_image(img)

# Definir un rectángulo (se ajustará automáticamente si es demasiado grande)
x1, y1 = 50, 60
x2, y2 = 150, 160
suma = region_sum(I, x1, y1, x2, y2)
print(f"Suma de intensidades en el rectángulo ({x1},{y1}) a ({x2},{y2}): {suma}")

# Mostrar resultados
plt.subplot(1,2,1)
plt.title("Imagen original")
plt.imshow(img, cmap="gray")

plt.subplot(1,2,2)
plt.title("Imagen integral")
plt.imshow(I, cmap="gray")
plt.show()
