
from scipy.ndimage import generic_filter
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from numba import jit, njit
from types import SimpleNamespace

import time
import h5py
import numpy as np
import os
import xarray
import rioxarray  # se debe importar para usar rio desde datasets de xarray
import rasterio
import matplotlib
import matplotlib.pyplot as plt


# apt install python3-tk
matplotlib.use('TkAgg')


# OBS:
# La opción fastmath=True hace que focal asigne nan a más píxeles de los deseados en focal (ejemplos al final)


@njit
def custom_mean_idic(arr):
    if np.isnan(arr[4]) and not np.isnan(np.delete(arr, 4)).all():
        return np.nanmean(np.delete(arr, 4))
    else:
        return arr[4]


@njit
def custom_mean_smap(arr):
    central_value = arr[len(arr) // 2]
    if np.isnan(central_value):
        return np.nan
    else:
        return np.nanmean(arr)


@njit
def focal_1(arr, size):
    """Calculate the focal mean of `arr` using a window of `size`."""
    # Create an empty array to store the result
    result = np.empty_like(arr)
    result[:] = np.nan
    # Iterate over each pixel in the input array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Compute the range of pixels to use for the focal operation
            start_i = max(i - size // 2, 0)
            end_i = min(i + size // 2 + 1, arr.shape[0])
            start_j = max(j - size // 2, 0)
            end_j = min(j + size // 2 + 1, arr.shape[1])
            # Compute the mean of the subset of pixels
            subset = arr[start_i:end_i, start_j:end_j].flatten()
            # result[i, j] = custom_mean_smap(subset)
            if not np.isnan(subset[len(subset) // 2]):
                result[i, j] = np.nanmean(subset)
    return result


@njit
def focal_2(arr, size):
    """Calculate the focal mean of `arr` using a window of `size`."""
    # Create an empty array to store the result
    result = np.empty_like(arr)
    result[:] = np.nan
    # Iterate over each pixel in the input array
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Compute the range of pixels to use for the focal operation
            start_i = max(i - size // 2, 0)
            end_i = min(i + size // 2 + 1, arr.shape[0])
            start_j = max(j - size // 2, 0)
            end_j = min(j + size // 2 + 1, arr.shape[1])
            # Compute the mean of the subset of pixels
            if not np.isnan(arr[i, j]):
                subset = arr[start_i:end_i, start_j:end_j].flatten()
                result[i, j] = np.nanmean(subset)
    return result


@jit(forceobj=True, parallel=True)
def fasted_generic_filter(values):
    return generic_filter(
        input=values,
        function=custom_mean_smap,
        footprint=np.ones((5, 5)),
        mode='constant')


def plot_array_am(array_am, title='Título'):
    plt.imshow(array_am, cmap='viridis')  # Plot the array as a heatmap
    plt.colorbar()  # Add a colorbar legend
    plt.title(title)  # Add a title
    plt.show()  # Display the plot


if __name__ == '__main__':
    graficar = False

    # Leer archivos hdf5
    soil_moistures = []
    for h5_file in [x for x in os.listdir('test_files') if x != ".gitkeep"]:
        data = h5py.File(f"test_files/{h5_file}", 'r')
        c_soil_moisture = np.array(data['Soil_Moisture_Retrieval_Data_AM']['soil_moisture'])  # Get data from h5_file
        c_soil_moisture[c_soil_moisture == -9999] = np.nan  # Asignan NaN a -9999
        soil_moistures.append(c_soil_moisture)  # Add array to soil_moistures
    # Unir todos los arrays en uno solo usando para ello el promedio. OBS: cuando la misma celda, en todos los
    # rasters es NaN, se lanza el warning "Mean of empty slice", pero esto no implica que el array tiene solo NaNs
    soil_moisture = np.nanmean(soil_moistures, axis=0) if len(soil_moistures) > 1 else soil_moistures[0]

    # Calcular media y graficar
    print(f"media: {np.nanmean(soil_moisture)}")
    plot_array_am(soil_moisture, "Datos Originales") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_1_1 = generic_filter(soil_moisture, custom_mean_smap, size=5, mode='constant')
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_1_1)} -- generic_filter con size")
    plot_array_am(array_am_1_1, "generic_filter con size") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_1_2 = generic_filter(soil_moisture, custom_mean_smap, footprint=np.ones((5, 5)), mode='constant')
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_1_2)} -- generic_filter con footprint")
    plot_array_am(array_am_1_2, "generic_filter con footprint") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_1_3 = fasted_generic_filter(soil_moisture)
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_1_3)} -- generic_filter IDIC")
    plot_array_am(array_am_1_3, "generic_filter IDIC") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_2_1 = focal_1(soil_moisture, size=5)
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_2_1)} -- focal manual 1")
    plot_array_am(array_am_2_1, "focal manual 2") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_2_2 = focal_2(soil_moisture, size=5)
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_2_2)} -- focal manual 2")
    plot_array_am(array_am_2_2, "focal manual 2") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_3 = median_filter(soil_moisture, size=5, mode='wrap')
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_3)} -- median_filter")
    plot_array_am(array_am_3, "median_filter") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    # Define a 5x5 smoothing kernel
    kernel = np.ones((5, 5)) / 25
    # Apply the kernel to the array using convolution
    array_am_4 = convolve2d(soil_moisture, kernel, mode='same', boundary='wrap')
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_4)} -- convolve2d")
    plot_array_am(array_am_4, "convolve2d") if graficar else None

    #
    # Aplicar suavizado con xarray.rolling
    #

    # Definir EXTENT.
    EXTENT = SimpleNamespace()
    setattr(EXTENT, 'xmin', -180.0)
    setattr(EXTENT, 'xmax', 180.0)
    setattr(EXTENT, 'ymin', -85.044)
    setattr(EXTENT, 'ymax', 85.044)
    # OBS:
    #  - Los valores de los elementos del EXTENT se obtuvieron aquí:
    #    https://nsidc.org/data/spl3smap/versions/3

    # Ancho y alto de la ventana móvil
    ROLLING_WINDOW_SMOOTHING = 5

    # Convertir datos en archivos hdf5, a datasets xarray
    start = time.time()
    # definir transform
    transform = rasterio.transform.from_bounds(
        west=EXTENT.xmin, south=EXTENT.ymin, east=EXTENT.xmax, north=EXTENT.ymax,
        width=len(soil_moisture[0]), height=len(soil_moisture))
    # definir coordenadas de los valores extraídos del hdf5
    res_x = (abs(EXTENT.xmin) + abs(EXTENT.xmax))/len(soil_moisture[0])  # 0.09336099585062241
    xrange = np.arange(EXTENT.xmin, EXTENT.xmax, res_x if EXTENT.xmin < EXTENT.xmax else res_x * -1)
    res_y = (abs(EXTENT.ymin) + abs(EXTENT.ymax))/len(soil_moisture)
    yrange = np.arange(EXTENT.ymax, EXTENT.ymin, res_y if EXTENT.ymax < EXTENT.ymin else res_y * -1)
    coords = dict(latitude=yrange, longitude=xrange)
    # crear objeto xarray
    raster = xarray.DataArray(data=soil_moisture, coords=coords)\
        .astype('float32')\
        .rio.write_transform(transform)\
        .rio.write_crs('epsg:4326')
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(raster.values)} -- Conversión a XARRAY")

    # Calcular media y graficar
    print(f"media: {np.nanmean(raster.values)}")
    plot_array_am(raster.values, "Datos Originales en XARRAY") if graficar else None

    # Aplicar suavizado (similar a raster::focal)
    start = time.time()
    array_am_5 = raster.rolling(latitude=ROLLING_WINDOW_SMOOTHING,
                                longitude=ROLLING_WINDOW_SMOOTHING,
                                min_periods=7, center=True).mean()
    end = time.time()
    print(f"tiempo: {end - start} -- media: {np.nanmean(array_am_5.values)} -- xarray.rolling")
    plot_array_am(array_am_5.values, "xarray.rolling") if graficar else None

    parar = True

# Celdas en las que se puede observar el problema con fastmath
#
# i, j = 838, 1228
# start_i = max(i - size // 2, 0)
# end_i = min(i + size // 2 + 1, arr.shape[0])
# start_j = max(j - size // 2, 0)
# end_j = min(j + size // 2 + 1, arr.shape[1])
# result[start_i:end_i, start_j:end_j] = random_values
# result[i, j] = 0.9
# i, j = 848, 1225
# start_i = max(i - size // 2, 0)
# end_i = min(i + size // 2 + 1, arr.shape[0])
# start_j = max(j - size // 2, 0)
# end_j = min(j + size // 2 + 1, arr.shape[1])
# result[start_i:end_i, start_j:end_j] = random_values
# result[i, j] = 0.9
