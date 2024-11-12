# Ejercicio 2 - Inciso a)

## Investigacoón de CUDA-C

CUDA son las siglas de Comput Unified Device Architecture in C, es una plataforma de programación paralela y un modelo de computación desarrollado por NVIDA, diseñado para aprovechar la capacidad de procesamiento de las unidades de procesamiento gráfico (GPU). Esto permite acelerar tareas complejas como simulaciones, análisis de grandes volúmenes de datos o inteligencia artificial. Utiliza un modelo de programación paralelo que permite ejecutar múltiples operaciones a la vez, lo que mejora considerablemente el rendimiento en tareas que requieren gran capacidad de procesamiento. Se programa en C/C++ con extensiones específicas para manejar la GPU de manera eficiente.

## Ejemplo de suma de dos vectores utilizando CUDA-C

### 1. Función que se ejecuta en la GPU
```cpp
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

```
Es una función llamda addKernel. Se va a ejecutar en la GPU. Toma dos listas de números (a y b), las suma elemento por elemento y guarda el resultado en una tercera lista (c).

### 2. Función principal

```cpp
int main()
{
    const int N = 10;
    int a[N], b[N], c[N];

```
En esta función se definen 3 arrgelos de 10 números: a, b y c. a y b son los arreglos que queremos sumar y c es donde se guardarán los datos.

