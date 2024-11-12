import scrapy

class BookItem(scrapy.Item):
    title = scrapy.Field()
    price = scrapy.Field()
    category = scrapy.Field()
    description = scrapy.Field()
    stock = scrapy.Field()

    
