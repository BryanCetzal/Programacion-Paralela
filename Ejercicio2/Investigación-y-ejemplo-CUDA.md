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
* Es una función llamda addKernel. Se va a ejecutar en la GPU. Toma dos listas de números (a y b), las suma elemento por elemento y guarda el resultado en una tercera lista (c).

### 2. Función principal

```cpp
int main()
{
    const int N = 10;
    int a[N], b[N], c[N];

```
* En esta función se definen 3 arrgelos de 10 números: a, b y c. a y b son los arreglos que queremos sumar y c es donde se guardarán los datos.

### 3. Inicialización de los vectores a  y b

```cpp
for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
}

```
### 4. Reserva de memoria en la GPU

```cpp

int *d_a, *d_b, *d_c;
cudaMalloc((void**)&d_a, N * sizeof(int));
cudaMalloc((void**)&d_b, N * sizeof(int));
cudaMalloc((void**)&d_c, N * sizeof(int));

```

* cudaMalloc() asigna memoria en la GPU.
* d_a, d_b y d_c son punteros en la GPU donde se almacenarán las copias de los vectores a, b y c.

### 5. Copia de los datos de la CPU a la GPU

```cpp

cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

```

* cudaMemcpy() copia los datos de la memoria de la CPU a la memoria de la GPU.
* cudaMemcpyHostToDevice es la dirección de la copia, indicando que es de CPU a GPU.


