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
