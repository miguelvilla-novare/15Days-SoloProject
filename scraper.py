from selenium import webdriver

try:
    driver = webdriver.Chrome()  # Or specify path if you *must*
    driver.get("https://www.google.com")
    print("ChromeDriver is working!")
    driver.quit()
except Exception as e:
    print(f"Error: {e}")