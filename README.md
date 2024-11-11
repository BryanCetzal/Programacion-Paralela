# Programacion-Paralela
Ejercicios de práctica en programación paralela  

## Ejercicio 1
**Caso de Uso:** Monitoreo de precios de competencia para una tienda de comercio electrónico

**Contexto**  
Una tienda en linea de venta de libros (LibrosU MT.com) enfrenta el reto de mantenerse competitiva en un mercado dinámico y saturado. Para optimizar su estrategia de precios, es necesario monitorear los precios de productos similares en las plataformas de su principal competidor. Este proceso es esencial para ajustar precios en tiempo reat, aprovechar oportunidades de venta y maximizar márgenes de ganancia. Sin embargo, realizar este monitoreo manualmente es inviable debido a la gran cantidad de prod uctos y la alta frecuencia de cambios de precios en el mercado.

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
