import requests
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse

def download_images_with_basic_auth(url, username, password, output_folder="downloaded_images_auth"):
    """
    Downloads images from href attributes on a webpage requiring basic HTTP authentication.

    Args:
        url (str): The URL of the webpage to scrape.
        username (str): The username for authentication.
        password (str): The password for authentication.
        output_folder (str): The folder to save the downloaded images to.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Make the initial request with basic authentication
        response = requests.get(url, auth=HTTPBasicAuth(username, password))
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage {url} with authentication: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    links = soup.find_all(['img', 'a'])

    downloaded_count = 0
    for link in links:
        image_url = None
        if link.name == 'img' and 'src' in link.attrs:
            image_url = link['src']
        elif link.name == 'a' and 'href' in link.attrs:
            href = link['href']
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp')
            if href.lower().endswith(image_extensions):
                image_url = href

        if image_url:
            absolute_image_url = urljoin(url, image_url)
            parsed_url = urlparse(absolute_image_url)
            image_filename = os.path.basename(parsed_url.path)
            
            if not image_filename or '.' not in image_filename:
                image_filename = f"image_{downloaded_count}{os.path.splitext(parsed_url.path)[1] or '.jpg'}"
            
            save_path = os.path.join(output_folder, image_filename)

            try:
                # Download the image with authentication (requests session will handle it)
                img_data = requests.get(absolute_image_url, auth=HTTPBasicAuth(username, password), stream=True)
                img_data.raise_for_status()

                with open(save_path, 'wb') as f:
                    for chunk in img_data.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {image_filename}")
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {absolute_image_url} with authentication: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with {absolute_image_url}: {e}")

    print(f"\nFinished downloading. Total images downloaded: {downloaded_count}")

# Example usage for Basic Authentication:
if __name__ == "__main__":
    target_url_basic_auth = "http://103.62.239.78:8081/remote/IFPRI_Images/paddy/"  # A test URL for basic auth
    my_username = "bkc"
    my_password = "agro$0923"
    download_images_with_basic_auth(target_url_basic_auth, my_username, my_password, "/home/dc/cntk/wheatcam5/")

    # IMPORTANT: Replace with your actual URL, username, and password for real-world scenarios.
    # e.g., target_url_basic_auth
