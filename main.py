import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import random
import re
import sys

# Configure console encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # For Python < 3.7
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# URLs for different categories
URL_WOMEN = "https://boulder-top.com/comp/bss25/page/ranking/r=62&k=153&v=42&c=53&h="  # k=153 for Frauen (women)
URL_MEN = "https://boulder-top.com/comp/bss25/page/ranking/r=62&k=152&v=42&c=53&h="    # k=152 for Männer (men)

def safe_print(text):
    """Print text safely, handling Unicode characters."""
    try:
        print(text)
    except UnicodeEncodeError:
        # If encoding fails, try to encode with 'utf-8' and replace unmappable characters
        print(text.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))

def get_optimized_chrome_options():
    """Return optimized Chrome options for Selenium."""
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--disable-features=VizDisplayCompositor')
    options.add_experimental_option('prefs', {
        'profile.managed_default_content_settings.images': 2,
        'profile.managed_default_content_settings.stylesheets': 2,
        'profile.managed_default_content_settings.cookies': 2,
        'profile.managed_default_content_settings.javascript': 1,
        'profile.managed_default_content_settings.plugins': 2,
        'profile.managed_default_content_settings.popups': 2,
        'profile.managed_default_content_settings.geolocation': 2,
        'profile.managed_default_content_settings.notifications': 2,
        'profile.managed_default_content_settings.auto_select_certificate': 2,
        'profile.managed_default_content_settings.fullscreen': 2,
        'profile.managed_default_content_settings.mouselock': 2,
        'profile.managed_default_content_settings.mixed_script': 2,
        'profile.managed_default_content_settings.media_stream': 2,
        'profile.managed_default_content_settings.media_stream_mic': 2,
        'profile.managed_default_content_settings.media_stream_camera': 2,
        'profile.managed_default_content_settings.protocol_handlers': 2,
        'profile.managed_default_content_settings.ppapi_broker': 2,
        'profile.managed_default_content_settings.automatic_downloads': 2,
        'profile.managed_default_content_settings.midi_sysex': 2,
        'profile.managed_default_content_settings.push_messaging': 2,
        'profile.managed_default_content_settings.ssl_cert_decisions': 2,
        'profile.managed_default_content_settings.metro_switch_to_desktop': 2,
        'profile.managed_default_content_settings.protected_media_identifier': 2,
        'profile.managed_default_content_settings.app_banner': 2,
        'profile.managed_default_content_settings.site_engagement': 2,
        'profile.managed_default_content_settings.durable_storage': 2
    })
    return options

def fetch_page_selenium(url):
    """Fetch a web page using Selenium and return its HTML."""
    options = get_optimized_chrome_options()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    driver.get(url)
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'card-body'))
        )
    except Exception:
        print("Timeout waiting for card-body div to appear.")
    html = driver.page_source
    driver.quit()
    return html

def fetch_page_selenium_single(driver, url):
    """Fetch a single page using an existing Selenium driver."""
    driver.get(url)
    return driver.page_source

def parse_gym_details(soup_details):
    """Parse gym details from a participant's details page."""
    gyms = []
    col_divs = soup_details.find_all('div', class_='col-md-12')
    i = 0
    while i < len(col_divs):
        div = col_divs[i]
        if 'background: var(--dark)' in (div.get('style') or ''):
            gym_name = None
            h5 = div.find('h5')
            if h5:
                gym_name = h5.get_text(strip=True)
            climbs = []
            if i + 1 < len(col_divs):
                next_div = col_divs[i + 1]
                if 'background-color: #a9a9a9' in (next_div.get('style') or ''):
                    for climb_div in next_div.find_all('div', id=lambda x: x and x.startswith('div-')):
                        btn = climb_div.find('button')
                        number_div = climb_div.find('div')
                        climb_number = number_div.get_text(strip=True) if number_div else None
                        completed_gym = btn and 'btn-success' in btn.get('class', [])
                        if completed_gym:
                            climbs.append(climb_number)
            if gym_name:
                gym_match = re.match(r"(.+?):\s*(\d+)\s*/\s*(\d+)\s*-\s*(\d+)%", gym_name)
                if gym_match:
                    gym_struct = {
                        'gym': gym_match.group(1).strip(),
                        'completed': int(gym_match.group(2)),
                        'total': int(gym_match.group(3)),
                        'percent': int(gym_match.group(4)),
                        'completed_climbs': climbs
                    }
                else:
                    gym_struct = {
                        'gym': gym_name,
                        'completed': None,
                        'total': None,
                        'percent': None,
                        'completed_climbs': climbs
                    }
                gyms.append(gym_struct)
        i += 1
    return gyms

def parse_participants(html):
    """Parse the main ranking page and return a list of participants with their details links."""
    soup = BeautifulSoup(html, 'html.parser')
    card_bodies = soup.find_all('div', class_='card-body')
    participants = []
    for card in card_bodies:
        rankings = card.find_all('div', class_='ranking', style=True)
        for ranking in rankings:
            ranking_text = ranking.find('div', class_='ranking-text')
            if ranking_text:
                left = ranking_text.find('span', class_='ranking-left')
                details_link = None
                for a in ranking.find_all_next('a', href=True):
                    button = a.find('button')
                    if button and 'Details' in button.get_text():
                        details_link = a['href']
                        break
                name_rank_match = re.match(r"(\d+)\.\s*(.+?)(\d+\s*/\s*160)", left.get_text(strip=True) if left else '')
                if name_rank_match:
                    rank = int(name_rank_match.group(1))
                    name = name_rank_match.group(2).strip()
                    completed = int(name_rank_match.group(3).split('/')[0].strip())
                else:
                    rank = None
                    name = left.get_text(strip=True) if left else ''
                    completed = None
                participants.append({
                    'climber': name,
                    'rank': rank,
                    'completed': completed,
                    'details_link': details_link
                })
    return participants

def scrape_category(url, category_name):
    """Scrape participants for a specific category and return the data."""
    safe_print(f"🔍 Scraping {category_name}...")
    safe_print(f"🌐 URL: {url}")
    
    html = fetch_page_selenium(url)
    participants = parse_participants(html)
    total = len(participants)
    
    if total == 0:
        safe_print(f"❌ No participants found for {category_name}")
        return []
    
    safe_print(f"👥 Found {total} {category_name}. Starting detailed scraping...")
    
    options = get_optimized_chrome_options()
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
    all_results = []
    
    for idx, participant in enumerate(participants, 1):
        if not participant['details_link']:
            safe_print(f"[{idx}/{total}] ⚠️  Skipping {participant['climber']} (no details link)")
            continue
        
        safe_print(f"[{idx}/{total}] 🧗 Scraping {participant['climber']} (rank: {participant['rank']})...")
        
        try:
            details_html = fetch_page_selenium_single(driver, participant['details_link'])
            soup_details = BeautifulSoup(details_html, 'html.parser')
            gyms = parse_gym_details(soup_details)
            
            participant_data = {
                'climber': participant['climber'],
                'rank': participant['rank'],
                'completed': participant['completed'],
                'gyms': gyms
            }
            
            all_results.append(participant_data)
            
        except Exception as e:
            safe_print(f"❌ Error scraping {participant['climber']}: {e}")
            continue
        
        # Add random delay to be respectful to the server
        time.sleep(random.uniform(0.4, 1.2))
    
    driver.quit()
    safe_print(f"✅ Completed scraping {len(all_results)} {category_name}")
    return all_results

def main():
    """Main function to scrape both men and women and save to results.json"""
    safe_print("🚀 Starting Boulder Summer Sessions 2025 scraping...")
    
    # Scrape men's results
    safe_print("\n=== SCRAPING MEN'S RESULTS ===")
    men_data = scrape_category(URL_MEN, "male participants")
    
    # Scrape women's results  
    safe_print("\n=== SCRAPING WOMEN'S RESULTS ===")
    women_data = scrape_category(URL_WOMEN, "female participants")
    
    # Combine and save results
    safe_print("\n=== SAVING COMBINED RESULTS ===")
    combined_data = {
        "metadata": {
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_participants": len(men_data) + len(women_data),
            "men_count": len(men_data),
            "women_count": len(women_data)
        },
        "men": men_data,
        "women": women_data
    }
    
    try:
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        safe_print(f"🎉 Successfully saved results to results.json!")
        safe_print(f"   📊 Total participants: {len(men_data) + len(women_data)}")
        safe_print(f"   👨 Men: {len(men_data)}")
        safe_print(f"   👩 Women: {len(women_data)}")
        
    except Exception as e:
        safe_print(f"❌ Error saving results: {e}")

if __name__ == "__main__":
    main()
