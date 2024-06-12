import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException

# Define paths
input_folder = 'D:/neu_studium/Master_Semester_01/KI/Projekt/test/wildfire'  # Path to the folder with images named as coordinates
output_folder = 'D:/neu_studium/Master_Semester_01/KI/New_Satelite_Images'  # Path to the folder to save screenshots

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

try:
    # Set up Selenium
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')

    print("Initializing WebDriver")
    driver = webdriver.Chrome(options=options)
    print("WebDriver initialized successfully")

    # URL of the website to automate
    url = 'https://earthexplorer.usgs.gov/'
    driver.get(url)
    print("Browser setup and navigated to URL")

    def get_coordinates_from_filename(filename):
        """Extract coordinates from filename."""
        name, _ = os.path.splitext(filename)
        latitude, longitude = name.split(',')
        return latitude.strip(), longitude.strip()

    def click_add_coordinate_button(driver):
        """Click the Add Coordinate button, ensuring it is visible and retrying if necessary."""
        try:
            add_coordinate_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.ID, "coordEntryAdd"))
            )
            driver.execute_script("arguments[0].scrollIntoView();", add_coordinate_button)
            add_coordinate_button.click()
            time.sleep(2)
        except Exception as e:
            print(f"First attempt to click Add Coordinate button failed: {e}")
            driver.save_screenshot(os.path.join(output_folder, 'add_coordinate_click_failed.png'))
            # Retry clicking the button using JavaScript
            try:
                driver.execute_script("arguments[0].click();", add_coordinate_button)
                time.sleep(2)
                print("Clicked Add Coordinate button using JavaScript on second attempt")
            except Exception as e2:
                print(f"Second attempt to click Add Coordinate button also failed: {e2}")
                driver.save_screenshot(os.path.join(output_folder, 'add_coordinate_click_failed_js.png'))

    def automate_screenshot(latitude, longitude, output_path, width=1920, height=1080):
        """Automate the process of taking a screenshot."""
        try:
            print(f"Processing coordinates: Latitude={latitude}, Longitude={longitude}")

            # Click the Polygon button
            print("Waiting for the Polygon button to be clickable")
            polygon_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.ID, "tabPolygon"))
            )
            polygon_button.click()
            print("Clicked the Polygon button")
            driver.save_screenshot(os.path.join(output_folder, 'step_polygon_click.png'))

            # Click the Decimal button
            print("Waiting for the Decimal button to be clickable")
            decimal_button = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.ID, "latlonfmtdec"))
            )
            decimal_button.click()
            print("Clicked the Decimal button")
            driver.save_screenshot(os.path.join(output_folder, 'step_decimal_click.png'))

            # Click the Add Coordinate button
            print("Clicking the Add Coordinate button")
            click_add_coordinate_button(driver)

            # Wait for the popup to appear
            popup_visible = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.ui-dialog"))
            )

            if popup_visible:
                print("Popup is visible")
                driver.save_screenshot(os.path.join(output_folder, 'popup_visible.png'))

                # Enter latitude and longitude
                print("Waiting for the Latitude input field to be present")
                latitude_input = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "latitude"))
                )

                print("Waiting for the Longitude input field to be present")
                longitude_input = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.ID, "longitude"))
                )

                print("Entering latitude using JavaScript")
                driver.execute_script("arguments[0].value = arguments[1];", latitude_input, latitude)
                print("Entered latitude")

                print("Entering longitude using JavaScript")
                driver.execute_script("arguments[0].value = arguments[1];", longitude_input, longitude)
                print("Entered longitude")

                driver.save_screenshot(os.path.join(output_folder, 'step_enter_coordinates.png'))

                # Click the Add button using JavaScript
                print("Clicking the Add button using JavaScript")
                add_button = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "button.ui-button.ui-corner-all.ui-widget"))
                )
                driver.execute_script("arguments[0].click();", add_button)
                print("Clicked the Add button")
                driver.save_screenshot(os.path.join(output_folder, 'step_add_click.png'))

                # Wait for the map to update (adjust time as necessary)
                time.sleep(5)
                print("Waited for the map to update")
                driver.save_screenshot(os.path.join(output_folder, 'step_map_update.png'))

                # Set the browser window size for consistent screenshots
                driver.set_window_size(width, height)
                print("Set the browser window size")

                # Take a screenshot
                screenshot_path = os.path.join(output_folder, f'{longitude},{latitude}.jpg')
                driver.save_screenshot(screenshot_path)
                print(f"Saved screenshot to {screenshot_path}")
            else:
                print("Popup did not become visible")
                driver.save_screenshot(os.path.join(output_folder, 'popup_not_visible.png'))

        except (ElementNotInteractableException, TimeoutException) as e:
            print(f"An error occurred during automation: {e}")
            driver.save_screenshot(os.path.join(output_folder, 'error_screenshot.png'))

    # Iterate over images in the input folder
    for image_filename in os.listdir(input_folder):
        if image_filename.endswith('.jpg'):  # Ensure you're only processing .jpg files
            latitude, longitude = get_coordinates_from_filename(image_filename)
            print(f"Latitude: {latitude}, Longitude: {longitude}")
            output_path = os.path.join(output_folder, image_filename)
            automate_screenshot(latitude, longitude, output_path)

except Exception as e:
    print(f"An error occurred during setup: {e}")

finally:
    # Close the browser
    if 'driver' in locals():
        driver.quit()
        print("Browser closed")
