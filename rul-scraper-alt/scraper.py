import requests
from bs4 import BeautifulSoup
import time
import os
from tqdm import tqdm

def idFromLink(link):
	idStart = link.index('id=') + 3
	idEnd = link[idStart:].index('&')
	return link[idStart:idStart + idEnd]

startPage, endPage = 100, 200 # Each page has 10 thesis
requestRate = 20 # Requests per second
reportAbstractFails = False

baseUrl = 'https://repozitorij.uni-lj.si/'
browseUrl = baseUrl	+ 'Brskanje2.php?kat1=jezik&kat2=1060&page=%s'

if not os.path.isdir('sl'):
	print('Directory must contain folder \'sl\'')
	exit(1)

if not os.path.isdir('en'):
	print('Directory must contain folder \'en\'')
	exit(1)

for page in tqdm(range(startPage, endPage)):
	response = requests.get(browseUrl % page)
	
	if response.status_code != 200:
		print('Invalid browse response')
		print(response)
		exit(1)
	
	soup = BeautifulSoup(response.text, 'html.parser')
	results = soup.select('.ZadetkiIskanja > tbody .Besedilo > a:first-of-type')
	links = [result["href"] for result in results]
	
	time.sleep(1/requestRate)
	for link in links:
		response = requests.get(baseUrl + link)

		if response.status_code != 200:
			print('Invalid link response')
			print(response)
			exit(1)

		try:
			soup = BeautifulSoup(response.text, 'html.parser')
			sl, en = soup.select('.izpisPovzetka, .izpisPovzetka_omejeno')
		except:
			if reportAbstractFails:
				print('Couldn\'t obtain abstract', link)
			continue

		sl = sl.text.replace('\r\n', '')
		en = en.text.replace('\r\n', '')

		id = idFromLink(link)
		with open("sl/%s.txt" % id, "w", encoding="utf8") as file:
			file.write(sl)
		with open("en/%s.txt" % id, "w", encoding="utf8") as file:
			file.write(en)
		
		time.sleep(1/requestRate)