from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import os

# Setup
options = Options()
options.add_argument("--headless")  # so it runs without opening a browser window
driver = webdriver.Chrome(options=options)

base_url = "https://en.wikipedia.org/wiki/Naruto:_Shippuden_season_{}"
output_dir = "naruto_pdfs"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 23):
    url = base_url.format(i)
    driver.get(url)
    time.sleep(2)

    # Find and click the "Download as PDF" link
    try:
        pdf_link = driver.find_element("link text", "Download as PDF")
        pdf_link.click()
        time.sleep(5)

        # On the new page, click the actual "Download" button
        download_button = driver.find_element("link text", "Download")
        pdf_url = download_button.get_attribute("href")

        # Use requests to download the file
        import requests

        response = requests.get(pdf_url)
        with open(os.path.join(output_dir, f"Naruto_Season_{i}.pdf"), "wb") as f:
            f.write(response.content)
        print(f"Downloaded Naruto Season {i}")
    except Exception as e:
        print(f"Failed on Season {i}: {e}")

driver.quit()
