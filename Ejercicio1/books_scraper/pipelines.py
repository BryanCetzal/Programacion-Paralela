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

