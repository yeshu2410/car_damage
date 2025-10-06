#!/usr/bin/env python3
"""
Bing Image Scraper for Damaged Cars Dataset
Downloads 3000 images of damaged/crashed cars from Bing Images
"""

import os
import time
import hashlib
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import json
import urllib.parse

# Configuration
SEARCH_QUERIES = [
    "damaged car",
    "crashed car", 
    "wrecked vehicle",
    "car accident damage",
    "collision damage car",
    "vehicle body damage",
    "dented car",
    "scratched car bumper"
]
TARGET_COUNT = 3000
SAVE_DIR = "damaged_cars"
MIN_IMAGE_SIZE = (200, 200)

os.makedirs(SAVE_DIR, exist_ok=True)


def setup_driver():
    """Setup Chrome driver with options"""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    os.environ['WDM_LOCAL'] = '1'
    os.environ['WDM_LOG_LEVEL'] = '0'
    
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts
        )
    except Exception as e:
        print(f"Error: {e}")
        driver = webdriver.Chrome(options=opts)
    
    return driver


def scrape_bing_images(driver, query, max_images=500):
    """Scrape images from Bing"""
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.bing.com/images/search?q={encoded_query}&first=1"
    
    driver.get(url)
    time.sleep(3)
    
    image_urls = set()
    scroll_pause = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    
    while len(image_urls) < max_images and scroll_attempts < 30:
        # Find image thumbnails
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "a.iusc")
        
        for thumb in thumbnails:
            try:
                m_json = thumb.get_attribute("m")
                if m_json:
                    data = json.loads(m_json)
                    img_url = data.get("murl") or data.get("turl")
                    if img_url and img_url.startswith("http"):
                        image_urls.add(img_url)
                        if len(image_urls) >= max_images:
                            break
            except Exception:
                continue
        
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                see_more = driver.find_element(By.CSS_SELECTOR, ".btn_seemore")
                see_more.click()
                time.sleep(3)
            except:
                scroll_attempts += 1
        
        last_height = new_height
        scroll_attempts += 1
    
    return list(image_urls)


def download_image(url, save_path):
    """Download and save an image"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        
        img = Image.open(BytesIO(r.content))
        
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        if img.size[0] < MIN_IMAGE_SIZE[0] or img.size[1] < MIN_IMAGE_SIZE[1]:
            return False
        
        if img.size[0] > 2000 or img.size[1] > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        
        img.save(save_path, 'JPEG', quality=90)
        return True
    except Exception:
        return False


def main():
    """Main execution"""
    print("=" * 60)
    print("Bing Image Scraper - Damaged Cars Dataset")
    print("=" * 60)
    print(f"Target: {TARGET_COUNT} images")
    print(f"Save directory: {SAVE_DIR}\n")
    
    driver = setup_driver()
    all_urls = set()
    
    images_per_query = TARGET_COUNT // len(SEARCH_QUERIES) + 100
    
    for query in SEARCH_QUERIES:
        print(f"Searching: '{query}'...")
        try:
            urls = scrape_bing_images(driver, query, images_per_query)
            all_urls.update(urls)
            print(f"  Found {len(urls)} images")
            
            if len(all_urls) >= TARGET_COUNT:
                break
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    driver.quit()
    
    print(f"\nTotal unique URLs: {len(all_urls)}")
    print("Downloading images...\n")
    
    downloaded = 0
    url_list = list(all_urls)[:TARGET_COUNT + 500]
    
    for i, url in enumerate(tqdm(url_list, desc="Downloading", ncols=80)):
        if downloaded >= TARGET_COUNT:
            break
        
        sha = hashlib.sha1(url.encode()).hexdigest()[:8]
        filename = f"damaged_car_{downloaded:04d}_{sha}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        
        if os.path.exists(save_path):
            downloaded += 1
            continue
        
        if download_image(url, save_path):
            downloaded += 1
    
    print(f"\n{'=' * 60}")
    print(f"✅ Download complete!")
    print(f"{'=' * 60}")
    print(f"Successfully downloaded: {downloaded} images")
    print(f"Images saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()
