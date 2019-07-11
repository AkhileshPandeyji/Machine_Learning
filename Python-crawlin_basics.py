import bs4
import urllib.request as url


#static data parsing and crawling

path = 'https://www.flipkart.com/'
http = url.urlopen(path)

page = bs4.BeautifulSoup(http)

title = page.find('a',class_='_2cLu-1')
print(title.text)

titles = page.find_all('a',class_='_2cLu-1')

for item in titles:
    print(item.text)





#dynamic data parsing and crawling

moviename = input("Enter The name of the movie?:")

path_1 = 'https://www.imdb.com/find?red_=nv_sr_fn&q='

http_response = url.urlopen(path+moviename)
page = bs4.BeautifulSoup(http_response)


td  = page.find('td',class_='result_text')
#tds = page.find_all('td',class_='result_text') //list

a = td.find('a')

print(a.attrs)

link_part = a.attrs['href']

path_2 = 'https://www.imdb.com/title'+link_part

http_response = url.urlopen(path_2)

page_2 = bs4.BeautifulSoup(http_response)

div_2 = page_2.find('div',class_='title-wrapper')

print(div_2)

t = div_2.text.split()
#argument is delimitter

new_title = " ".join(t)

links = page_2.find_all('a',class_='quicklink')

reviewHref =links[2]['href']
url2 = 'https://www.imdb.com/'+reviewHref

http_response = url.urlopen(url2)
page_3 = bs4.BeautifulSoup(http_response)

reviews = page_3.find_all('a',class_='title')

print(reviews)


rating = page_3.find_all('span',class_='rating-other-user-rating')
span = rating[0].find('span')

usersdata ={}
users =[]
review = []
ratings = []
for i in range(len(reviews)):
    users.append('users_{}'.format(i))
    review.append(reviews[i].find().text)
    ratings.append(rating[i].text)
    usersdata['users']=users
    usersdata['reviews'] = review
    usersdata['ratings'] = ratings

import pandas as pd

dframe = pd.DataFrame(usersdata)

print(dframe)














