# Scrapy
Este proyecto es un scraper para el sitio Books to Scrape, que permite extraer información de libros como título, precio, disponibilidad y categoría, utilizando Scrapy y concurrent en Python.  

Para la realización de este ejercicio, se opto por usar **Scrapy** un framework que nos permite realizar el scraping de manera sencilla y nos permite configurar la paralelización desde sus archivos.

### **1. Instalación del framework y de la bd**
```bash
pip install scrapy
```
Para comprobar que la instalación es correcta desde la consola podemos poner scrapy y nos debe mostrar el menú de las opciones disponibles.

### **2. Creación de entorno**
```bash
scrapy startproject <nombre_del_proyecto>
```
Después de ejecutar el código anterior, se crea una carpeta con los archivos necesarios para usar el framework de scrapy. 
![](EstrucutraInicial.jpg)

### **3. Configuración del archivo items**
En nuestro caso nos interesan ciertas cosas en especifico, y en este archivo los definiremos
```python
import scrapy

class BookItem(scrapy.Item):
    title = scrapy.Field()
    price = scrapy.Field()
    category = scrapy.Field()
    description = scrapy.Field()
    stock = scrapy.Field()
```

### **4. Configuración de los settings de scrapy**  
Aquí se definen varias configuraciones importantes para el funcionamiento del scraper. A continuación, se explican algunas de las configuraciones presentes en el archivo:
- BOT_NAME: Define el nombre del bot que se utilizará en el proyecto. En este caso, es 'books_scraper'.
- REACTOR_THREADPOOL_MAXSIZE: Define el tamaño máximo del pool de hilos del reactor.

```python
BOT_NAME = 'books_scraper'

SPIDER_MODULES = ['books_scraper.spiders']
NEWSPIDER_MODULE = 'books_scraper.spiders'

# Respeta las reglas de robots.txt (cambia a False si quieres ignorarlas)
ROBOTSTXT_OBEY = True

# Configuración de concurrencia
CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 8
REACTOR_THREADPOOL_MAXSIZE = 20

# Pipeline de datos
ITEM_PIPELINES = {
   'books_scraper.pipelines.SQLitePipeline': 300,
}

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
```

### **5. Configuración de pipelines** 
Este archivo `pipelines.py` define una clase SQLitePipeline que se utiliza para procesar y almacenar los datos extraídos por el scraper en una base de datos SQLite
```python
import sqlite3
from concurrent.futures import ThreadPoolExecutor

class SQLitePipeline:
    def open_spider(self, spider):
        self.connection = sqlite3.connect('books.db')
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS books (
                title TEXT,
                price TEXT,
                category TEXT,
                description TEXT, 
                stock TEXT
            )
        ''')
        self.connection.commit()
        # Crear un pool de threads para procesar items en paralelo
        self.executor = ThreadPoolExecutor(max_workers=10)

    def close_spider(self, spider):
        self.executor.shutdown(wait=True)  # Esperar a que todos los threads terminen
        self.connection.close()

    def process_item(self, item, spider):
        # Usar un thread para procesar cada item
        self.executor.submit(self._save_item_to_db, item)
        return item

    def _save_item_to_db(self, item):
        # Crear una nueva conexión para cada thread
        conn = sqlite3.connect('books.db')
        cursor = conn.cursor()
        
        # Función para guardar el item en la base de datos
        cursor.execute('''
            INSERT INTO books (title, price, category, description, stock) VALUES (?, ?, ?, ?, ?)
        ''', (item['title'], item['price'], item['category'], item['description'], item['stock']))
        
        conn.commit()
        conn.close()
```

### **6. Configuración de spyder** 
Este archivo `books_spider.py` define una araña (spider) de Scrapy llamada BooksSpider que se utiliza para extraer información de libros del sitio web. A continuación, se detalla la funcionalidad de cada método en la clase BooksSpider:

**Clase BooksSpider**
  - `name:` Nombre de la araña, en este caso 'books_scraper'.
  - `start_urls:` Lista de URLs iniciales desde donde la araña comenzará a rastrear.
  
**Métodos**
```python
def parse(self, response):
        # Extraer todas las categorías
        categories = response.css('ul.nav-list ul li a::attr(href)').getall()
        for category_url in categories:
            yield response.follow(category_url, callback=self.parse_category)
```
- Extrae todas las categorías de libros del sitio web.
- Para cada URL de categoría, realiza una solicitud de seguimiento y llama al método `parse_category`.

```python
def parse_category(self, response):
        # Extraer la categoría actual
        category = response.css('div.page-header h1::text').get().strip()
        
        # Extraer enlaces de cada libro en la página de lista
        for book_link in response.css('article.product_pod div.image_container a::attr(href)').getall():
            # Hacer una solicitud de seguimiento para cada enlace de libro
            yield response.follow(book_link, callback=self.parse_book_details, meta={'category': category})

        # Navegar a la siguiente página de la lista de libros en la misma categoría, si existe
        next_page = response.css('li.next a::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse_category)
```
- Extrae el nombre de la categoría actual.
- Extrae los enlaces de cada libro en la página de la lista de libros.
- Para cada enlace de libro, realiza una solicitud de seguimiento y llama al método parse_book_details``, pasando la categoría como meta-dato.
- Navega a la siguiente página de la lista de libros en la misma categoría, si existe, y realiza una solicitud de seguimiento para continuar el rastreo.

```python
def parse_book_details(self, response):
        # Extraer la categoría desde meta
        category = response.meta['category']
        
        # Extraer detalles del libro en la página individual del libro
        item = BookItem()
        item['title'] = response.css('h1::text').get()
        item['price'] = response.css('.price_color::text').get()
        item['description'] = response.xpath('//article[contains(@class,"product_page")]/p/text()').get()
        item['category'] = category
        item['stock'] = response.css('.instock.availability::text').re_first(r'\((\d+) available\)')

        yield item

```
- Extrae la categoría desde los meta-datos.
- Extrae detalles del libro en la página individual del libro, incluyendo el título, precio y descripción.
- Crea un objeto `BookItem` y lo llena con los datos extraídos.

**Ejemplo de Uso**
Este spider se utiliza para rastrear el sitio books.toscrape.com, extrayendo información de libros como título, precio, descripción y categoría, y almacenándola en objetos BookItem para su posterior procesamiento.

**Notas**
- La araña sigue enlaces de categorías y libros, y navega por las páginas de la lista de libros dentro de cada categoría.
- Utiliza selectores CSS y XPath para extraer la información necesaria de las páginas web.

### **7. Ejecución del scraper**
Una vez configurado los archivos desde la terminal, colocados en la posición principal del scraper se ejecuta. 
```bash
scrapy crawl <nombre_del_spider>
```

### **8. Visualización de los datos obtenidos**
Para visualizar la base de datos generada, se puede instalar el siguiente [DB Browser para SQLite](https://download.sqlitebrowser.org/DB.Browser.for.SQLite-v3.13.1-win64.msi) para Windows.

Video Explicando el funcionamiento: [Scraper](youtube.com)
