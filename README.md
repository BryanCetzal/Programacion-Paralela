# Programacion-Paralela
Ejercicios de práctica en programación paralela  

## Ejercicio 1
**Caso de Uso:** Monitoreo de precios de competencia para una tienda de comercio electrónico

**Contexto**  
Una tienda en linea de venta de libros (LibrosUMT.com) enfrenta el reto de mantenerse competitiva en un mercado dinámico y saturado. Para optimizar su estrategia de precios, es necesario monitorear los precios de productos similares en las plataformas de su principal competidor. Este proceso es esencial para ajustar precios en tiempo reat, aprovechar oportunidades de venta y maximizar márgenes de ganancia. Sin embargo, realizar este monitoreo manualmente es inviable debido a la gran cantidad de prod uctos y la alta frecuencia de cambios de precios en el mercado.

**Objetivo**  
Automatizar la recopilación de precios de productos de competidores mediante web scraping y análisis de datos para ofrecer insights de ajustes en la estrategia de precios.

**Descripción del Proceso**  
1. Identificación de Competidores y Productos Clave:
El competidor directo de la empresa es (https://books.toscrape.com/index.html).

2. Configuración del Proceso de Web Scraping:
Desarrollar un script web scraping utilizando Python con paralelismo que extraiga precios y datos relevantes de cada producto (nombre, características principales, precios, descripción, etc.) en tiempo real. Puedes utilizar Scrapy o BeautifulSoup

3. Almacenamiento y Normalización de Datos:
Guardar los datos recopilados en una base de datos centralizada

**Proceso**
1. Iniciar Web Scraping -> 2. Extraer Datos de los competidores -> 3. Guardas los Datos.

## Ejercicio 2
Paralelismo dinámico con CUDA

a) Realiza una breve investigación y proporciona un ejemplo sencillo utilizando CUDA-C. El ejemplo debe ser ejectuable en Google Colab (https://colab.research.google.com). Asegúrate de incluir caputras de pantalla que demuestren la ejecución del código.

b) Desarrolla un programa utilizando paralelismo dinámico en CUDA que lea una imagen de 512 x 512 píxeles y la divida en 4 subimágenes. Cada subimagen, a su vez, se subdividirá en 4 subimágenes adicionales, generando un árbol cuaternario. Este proceso continuará recursivaemente hasta que cada hilo se encargue de una subimagen de 2 x 2 píxeles o hasta que se alcance el límite del sistema. El hilo calculará el promedio de los 4 píxeles y asignará ese valor a cada uno de ellos.

c) Realiza un vídeo relacionado al ejercicio anterior donde expliques el código que utilizaste y las imágenes de entrada y salida que obtuviste.

## Ejercicio 3

a) Configura Google Colab para ejecutar un ejemplo sencillo de openMP donde se despliegue el uso de 8 hilos. Asegúrate de incluir capturas de pantalla que demuestren la ejecución del código.

b) Realiza una base de datos con SQLite en el cual realices diversos querys en el cual insertes valores y realices consultas en paralelo utilizando Openmp en Google Colab. No olvides configurar SQLite para permitir escrituras concurrentes y medir el tiempo de ejecución del algoritmo paralelo en relación con el tiempo de ejecución secuencial.

c) Realiza un vídeo relacionado al ejercicio anterior donde expliques el códgio que utilizaste y la base de datos generada.

**Integrantes**
- [Emmanuel Cetzal](https://github.com/BryanCetzal/)
- [Luis Quintana](https://github.com/Luis-J-Quintana)
