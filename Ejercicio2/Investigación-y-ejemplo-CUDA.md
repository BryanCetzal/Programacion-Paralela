# Ejercicio 2 - Inciso a)

## Investigacoón de CUDA-C

CUDA son las siglas de Comput Unified Device Architecture in C, es una plataforma de programación paralela y un modelo de computación desarrollado por NVIDA, diseñado para aprovechar la capacidad de procesamiento de las unidades de procesamiento gráfico (GPU). Esto permite acelerar tareas complejas como simulaciones, análisis de grandes volúmenes de datos o inteligencia artificial. Utiliza un modelo de programación paralelo que permite ejecutar múltiples operaciones a la vez, lo que mejora considerablemente el rendimiento en tareas que requieren gran capacidad de procesamiento. Se programa en C/C++ con extensiones específicas para manejar la GPU de manera eficiente.

## Ejemplo de suma de dos vectores utilizando CUDA-C

```cpp
#include <iostream>
using namespace std;

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) C[index] = A[index] + B[index];
}
```
