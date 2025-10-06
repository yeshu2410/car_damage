#!/usr/bin/env python3
"""
Improved Google Images Scraper for Damaged Cars
Downloads images using multiple approaches and better error handling
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
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Configuration
QUERIES = [
    "damaged car",
    "crashed car", 
    "wrecked vehicle",
    "car accident damage",
    "collision car damage",
    "car body damage"
]
TARGET_COUNT = 3000
SAVE_DIR = "damaged_cars"
MIN_IMAGE_SIZE = (200, 200)

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("Google Images Scraper - Damaged Cars Dataset")
print("=" * 60)
print(f"Target: {TARGET_COUNT} images")
print(f"Save directory: {SAVE_DIR}\n")

# Setup Chrome driver
opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1920,1080")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")
opts.add_argument("--disable-dev-shm-usage")
opts.add_argument("--disable-blink-features=AutomationControlled")
opts.add_experimental_option("excludeSwitches", ["enable-automation"])
opts.add_experimental_option('useAutomationExtension', False)

os.environ['WDM_LOCAL'] = '1'
os.environ['WDM_LOG_LEVEL'] = '0'

print("Initializing Chrome WebDriver...")
try:
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts
    )
except Exception as e:
    print(f"Using system Chrome driver...")
    driver = webdriver.Chrome(options=opts)

# Collect URLs from multiple queries
all_urls = set()

for query_idx, query in enumerate(QUERIES, 1):
    if len(all_urls) >= TARGET_COUNT:
        break
    
    print(f"\n[{query_idx}/{len(QUERIES)}] Searching: '{query}'")
    
    try:
        url = f"https://www.google.com/search?tbm=isch&q={query.replace(' ', '+')}"
        driver.get(url)
        time.sleep(3)
        
        image_urls = set()
        last_height = 0
        scroll_attempts = 0
        max_scrolls = 30
        
        while len(image_urls) < (TARGET_COUNT // len(QUERIES)) and scroll_attempts < max_scrolls:
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Find all images with multiple selectors
            all_images = []
            selectors = [
                "img.rg_i",
                "img.YQ4gaf",
                "img.iPVvYb",
                "img[jsname]",
                "div[data-id] img"
            ]
            
            for selector in selectors:
                try:
                    images = driver.find_elements(By.CSS_SELECTOR, selector)
                    all_images.extend(images)
                except:
                    continue
            
            # Extract URLs
            for img in all_images:
                try:
                    # Try multiple attributes
                    src = (img.get_attribute("src") or 
                           img.get_attribute("data-src") or 
                           img.get_attribute("data-iurl"))
                    
                    if src and src.startswith("http") and not src.startswith("data:"):
                        # Filter out Google's static images
                        if "gstatic.com" not in src and "googleusercontent.com" in src:
                            image_urls.add(src)
                except:
                    continue
            
            # Click on thumbnails to get full URLs
            try:
                thumbnails = driver.find_elements(By.CSS_SELECTOR, "div.isv-r")[:20]
                for thumb in thumbnails:
                    if len(image_urls) >= (TARGET_COUNT // len(QUERIES)):
                        break
                    try:
                        thumb.click()
                        time.sleep(0.3)
                        
                        # Get full size image
                        full_imgs = driver.find_elements(By.CSS_SELECTOR, "img.n3VNCb, img.iPVvYb, img.sFlh5c")
                        for full_img in full_imgs:
                            try:
                                full_src = full_img.get_attribute("src")
                                if full_src and "googleusercontent.com" in full_src:
                                    image_urls.add(full_src)
                            except:
                                pass
                    except:
                        continue
            except:
                pass
            
            # Check scroll progress
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                # Try show more button
                try:
                    buttons = driver.find_elements(By.CSS_SELECTOR, ".mye4qd, .LZ4I, .r0zKGf")
                    for btn in buttons:
                        try:
                            btn.click()
                            time.sleep(2)
                            break
                        except:
                            continue
                except:
                    pass
                scroll_attempts += 1
            else:
                scroll_attempts = 0
            
            last_height = new_height
        
        all_urls.update(image_urls)
        print(f"  Collected {len(image_urls)} URLs (Total: {len(all_urls)})")
        
    except Exception as e:
        print(f"  Error: {e}")
        continue

driver.quit()

print(f"\n{'='*60}")
print(f"Total unique URLs collected: {len(all_urls)}")
print(f"{'='*60}\n")

if len(all_urls) == 0:
    print("❌ No URLs collected. Try running without --headless mode or check your internet connection.")
    exit(1)

# Download images
print("Starting download...\n")
downloaded = 0
failed = 0

for i, img_url in enumerate(tqdm(list(all_urls)[:TARGET_COUNT], desc="Downloading", ncols=80)):
    try:
        # Create filename
        sha = hashlib.sha1(img_url.encode()).hexdigest()[:8]
        filename = f"damaged_car_{i:04d}_{sha}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Skip if exists
        if os.path.exists(filepath):
            downloaded += 1
            continue
        
        # Download
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        r = requests.get(img_url, headers=headers, timeout=15, stream=True)
        r.raise_for_status()
        
        # Open and validate image
        img = Image.open(BytesIO(r.content))
        
        # Convert to RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Check minimum size
        if img.size[0] < MIN_IMAGE_SIZE[0] or img.size[1] < MIN_IMAGE_SIZE[1]:
            failed += 1
            continue
        
        # Resize if too large
        if img.size[0] > 4000 or img.size[1] > 4000:
            img.thumbnail((4000, 4000), Image.Resampling.LANCZOS)
        
        # Save
        img.save(filepath, 'JPEG', quality=95)
        downloaded += 1
        
    except Exception as e:
        failed += 1
        continue

print(f"\n{'='*60}")
print(f"✅ Download Complete!")
print(f"{'='*60}")
print(f"Successfully downloaded: {downloaded} images")
print(f"Failed: {failed}")
print(f"Images saved in: {SAVE_DIR}")
print()