#!/usr/bin/env python
# coding: utf-8

# # Procedures
soup.title
<title>Returns title tags and the content between the tags</title>

soup.title.string
#u'Returns the content inside a title tag as a string'

soup.p
<p class="title"><b>This returns everything inside the paragraph tag</b></p>

soup.p['class']
#u'className' (this returns the class name of the element)

soup.a
<a class="link" href="http://example.com/example" id="link1">This would return the first matching anchor tag</a>

We could use the find all, and return all the matching anchor tags
soup.find_all('a')
[<a class="link" href="http://example.com/example1" id="link1">link2</a>,
<a class="link" href="http://example.com/example2" id="link2">like3</a>,
<a class="link" href="http://example.com/example3" id="link3">Link1</a>]

soup.find(id="link3")
<a class="link" href="http://example.com/example3" id="link3">This returns just the matching element by ID</a>
# ## Web Scraping with BeautifulSoup

# In[1]:


import urllib.request as ur


# In[2]:


get_ipython().system(' pip install beautifulsoup4')


# In[4]:


from bs4 import BeautifulSoup as bs


# In[61]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


url = 'https://www.flipkart.com/laptops/~buyback-guarantee-on-laptops-/pr?sid=6bo%2Cb5g&uniqBStoreParam1=val1&wid=11.productCard.PMU_V2'


# In[5]:


page = ur.urlopen(url)


# In[6]:


soup = bs(page)


# In[106]:


products=[] #List to store name of the product
prices=[] #List to store price of the product
ratings=[] #List to store rating of the product
for a in soup.findAll('a', href=True, attrs={'class':'_31qSD5'}):
    product=a.find('div', attrs={'class':'_3wU53n'})
    price=a.find('div', attrs={'class':'_1vC4OE _2rQ-NK'})
    rating=a.find('div', attrs={'class':'hGSR34'})
    products.append(product.text)
    prices.append(price.text)
    ratings.append(rating.text)


# In[107]:


df = pd.DataFrame({'Product Name':products,'Price':prices,'Rating':ratings}) 
df.to_csv('products.csv', index=False, encoding='utf-8')


# In[108]:


file = pd.read_csv('products.csv')
file


# ## Web Scraping with Scrapy 1

# In[6]:


get_ipython().system(' pip install scrapy')


# In[11]:


import string
import scrapy
from scrapy import Request
from scrapy.crawler import CrawlerProcess


# In[25]:


class MangaBaseSpider(scrapy.Spider):
    name = "Manga" #Name of the spider
    start_urls = ['https://myanimelist.net/manga.php'] #URL that you start to crawl from

    def parse(self, response):
        xp = "//div[@id='horiznav_nav']//li/a/@href" #Selector of the menu
        return (Request(url, callback=self.parse_manga_list_page) for url in response.xpath(xp).extract())

    def parse_manga_list_page(self, response):
        for tr_sel in response.css('div.js-categories-seasonal tr ~ tr'): #Selector of the whole paragraph
            yield {

                "title":  tr_sel.css('a[id] strong::text').extract_first().strip(),

                "synopsis": tr_sel.css("div.pt4::text").extract_first(),

                "type_": tr_sel.css('td:nth-child(3)::text').extract_first().strip(),

                "episodes": tr_sel.css('td:nth-child(4)::text').extract_first().strip(), 

                "rating": tr_sel.css('td:nth-child(5)::text').extract_first().strip(),
            }

        next_urls = response.xpath("//div[@class='spaceit']//a/@href").extract()

        for next_url in next_urls:
            yield Request(response.urljoin(next_url), callback=self.parse_anime_list_page)


# In[27]:


scrapy crawl myspider -o data.csv


# In[ ]:


class BrickSetSpider(scrapy.Spider):
    name = 'brick_spider'
    start_urls = ['http://brickset.com/sets/year-2016']

    def parse(self, response):
        xp = "//div[@id='pagination']/ul//li/a/@href" #Selector of the menu
        return (Request(url, callback=self.parse_brick_list_page) for url in response.xpath(xp).extract())

    def parse_brick_list_page(self, response):
        for tr_sel in response.css('.set'): #Selector of the whole paragraph
            yield {

                "name":  tr_sel.css('h1 ::text').extract_first().strip(),

                "pieces": tr_sel.xpath('.//dl[dt/text() = "Pieces"]/dd/a/text()').extract_first(),

                "minifigs": tr_sel.xpath('.//dl[dt/text() = "Minifigs"]/dd[2]/a/text()').extract_first().strip(),

                "image": tr_sel.css('img ::attr(src)').extract_first().strip(), 
            }

        next_urls = response.css('.next a ::attr(href)').extract()

        for next_url in next_urls:
            yield Request(response.urljoin(next_url), callback=self.parse_brick_list_page)


# ## Web Scraping (Flight Data Kayak)

# In[40]:


import re
from datetime import date, timedelta, datetime
import time


# In[31]:


url_kayak = 'https://www.kayak.com/flights/LAX-SFO/2020-03-09/2020-03-12?sort=bestflight_a&fs=airlines=WN,DL,AN'


# In[32]:


page_source = ur.urlopen(url_kayak)


# In[37]:


soup = bs(page_source)


# In[ ]:


origin = "LAX"
destination = "SFO"
startdate = "202o-03-09"
enddate = "2020-03-12"
currency = "USD"

deptimes = soup.find_all('span', attrs={'class': 'depart-time base-time'})
arrtimes = soup.find_all('span', attrs={'class': 'arrival-time base-time'})
meridies = soup.find_all('span', attrs={'class': 'time-meridiem meridiem'})

deptime = []
for div in deptimes:
    deptime.append(div.getText()[:-1])    

arrtime = []
for div in arrtimes:
    arrtime.append(div.getText()[:-1])   

meridiem = []
for div in meridies:
    meridiem.append(div.getText())  

deptime = np.asarray(deptime)
deptime = deptime.reshape(int(len(deptime)/2), 2)

deptime_o = [m+str(n) for m,n in zip(deptime[:,0],meridiem[:,0])]
arrtime_d = [m+str(n) for m,n in zip(arrtime[:,0],meridiem[:,1])]
deptime_d = [m+str(n) for m,n in zip(deptime[:,1],meridiem[:,2])]
arrtime_o = [m+str(n) for m,n in zip(arrtime[:,1],meridiem[:,3])]


# In[45]:


regex = re.compile('Common-Booking-MultiBookProvider (.*)multi-row Theme-featured-large(.*)')
price_list = soup.find_all('div', attrs={'class': regex})

price = []
for div in price_list:
    price.append(int(div.getText().split('\n')[3][1:-1]))


# In[ ]:


df = pd.DataFrame({ "origin" : origin,
                    "destination" : destination,
                    "startdate" : startdate,
                    "enddate" : enddate,
                    "price": price,
                    "currency": currency,
                    "deptime_o": deptime_o,
                    "arrtime_d": arrtime_d,
                    "deptime_d": deptime_d,
                    "arrtime_o": arrtime_o,
                  })


# In[ ]:


df.to_csv('kayak.csv', index=False, encoding='utf-8')
file = pd.read_csv('kayak.csv')
file.head()


# In[ ]:


heatmap_results = pd.pivot_table(results_agg , values='price', index=['destination'], columns='startdate')

sns.set(font_scale=1.5)
plt.figure(figsize = (18,6))
sns.heatmap(heatmap_results, annot=True, annot_kws={"size": 24}, fmt='.0f', cmap="RdYlGn_r")

