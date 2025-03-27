import requests 
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import List
import logging 
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class CarInfo: 
	brand : str
	name : str
	images_url : List[str]
	# id : str

def get_soup(url: str, headers) -> BeautifulSoup:
	response = requests.get(url, headers=headers)
	# print(f"Mã trạng thái: {response.status_code}")
	return BeautifulSoup(response.content, 'html.parser')

def save_json_data(carInfo:CarInfo, filename:str):
	if Path(filename).exists():
		with open(filename, 'r', encoding="utf-8") as f:
			try: 
				existing_data = json.load(f)
			except json.JSONDecodeError:
				existing_data=[]
	else : 
		existing_data = []

	cars_dict = [vars(car) for car in carInfo]
	existing_data.extend(cars_dict)

	with open(filename, 'w', encoding="utf-8") as f:
		json.dump(existing_data, f, ensure_ascii=False, indent=2)

def extract_car_info(car_url:str, brand_name:str) -> CarInfo:
	soup = get_soup(car_url, headers)
	try :
		# print(car_url)
		name = soup.select_one('.listing-title').text.strip()
		images_url =[]
		gallery_images = (
			soup.select('img[modal-src]')
		)
		
		for idx, img in enumerate(gallery_images):
			src = img.get("src")
			if src not in images_url : 
				images_url.append(src)
			if idx == 3 : break
		return CarInfo(
			brand=brand_name,
			name=name,
			images_url=images_url,
		)
	except Exception as e : 
		logging.error(f"Error extracting car info from {url} : {str(e)}")

def main(url:str, headers:list):
	soup = get_soup(url, headers=headers)
	brand_section = soup.find("ul", class_="sds-list top-links")
	acura_brand = brand_section.find_all("a")
	for idx, acura_son in enumerate(acura_brand):
		brand_name = acura_son.text.strip()
		brand_href = acura_son.get("href") 

		if brand_href : 
			full_url = brand_href if brand_href.startswith("http") else f"{url.rstrip('/')}/{brand_href.lstrip('/')}"
			print(full_url)
		
		html_content = get_soup(full_url, headers=headers)
		brand_total = html_content.find_all("a", class_="vehicle-card-link")
		for idx_, brand_detail in enumerate(brand_total):
			car_detail_url = [urljoin(url, brand_detail.get("href"))]
			carInfo_total = []
			for car_brand in car_detail_url:
				car_info = extract_car_info(car_brand, brand_name)
				carInfo_total.append(car_info)
			save_json_data(carInfo_total, "dataset/raw_data.json")
			# if idx_ == 2 : break

		# break
	# if idx == 2 : break

### Main ###
if __name__ == '__main__' :
	url = "https://www.cars.com/shopping/"  # Thay bằng URL bạn muốn lấy dữ liệu
	headers = {
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
			'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Vivaldi/6.0.2980.60 Chrome/108.0.5359.170 Safari/537.36'
			}
	main(url, headers)
	requests.session()
