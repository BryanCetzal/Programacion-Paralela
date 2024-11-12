import scrapy
from books_scraper.items import BookItem

class BooksSpider(scrapy.Spider):
    name = "books_scraper"
    start_urls = ['http://books.toscrape.com/']

    def parse(self, response):
        # Extraer todas las categorías
        categories = response.css('ul.nav-list ul li a::attr(href)').getall()
        for category_url in categories:
            yield response.follow(category_url, callback=self.parse_category)

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

