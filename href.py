import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image

def get_all_hrefs(url, username="bkc", password="agro$0923", download_folder="/home/dc/cntk/myimg/"):
    """
    Fetches a webpage and extracts all href attribute values.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        list: A list of all unique href values found on the page.
    """
    try:
        response = requests.get(url, auth=(username, password))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    hrefs = set()  # Use a set to store unique hrefs

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Resolve relative URLs to absolute URLs
        absolute_href = urljoin(url, href)
        hrefs.add(absolute_href)
        with 
    # You might also want to find hrefs in other tags like <link>, <script>, etc.
    # For <link> tags (e.g., stylesheets)
    for link_tag in soup.find_all('link', href=True):
        href = link_tag['href']
        absolute_href = urljoin(url, href)
        hrefs.add(absolute_href)

    # For <script> tags (if they have a src attribute for external scripts)
    for script_tag in soup.find_all('script', src=True):
        src = script_tag['src']
        absolute_src = urljoin(url, src)
        hrefs.add(absolute_src)
    for link in hrefs:
     if link[-3:] == 'jpg':
        filepath = str(link)
        print(filepath)

    #return sorted(list(hrefs)) # Convert back to list and sort for consistent output

if __name__ == "__main__":
    # Example usage:
    target_url = "http://103.62.239.78:8081/remote/IFPRI_Images/IMGcrop_bkp_oct2019/" 

    print(f"Extracting href values from: {target_url}\n")
    all_links = get_all_hrefs(target_url, username="bkc", password="agro$0923", download_folder="/home/dc/cntk/myimg/")
     if all_links:
        print("Found the following href values:")
        for link in all_links:
            if link[-3:] == 'jpg':
             filepath = str(link)
             
            print(filepath)
             r = requests.get(filepath)
              print(r.status_code)
             print('ok....')
             #with open(filepath, 'wb') as f:
              #f.write(r.content)
              #print(f"Downloaded: {save_path}")
    else:
        print("No href values found or an error occurred.")

    # You can also save them to a file:
    # output_filename = "href_values.txt"
    # with open(output_filename, "w", encoding="utf-8") as f:
    #     for link in all_links:
    #         f.write(link + "\n")
    # print(f"\nHref values saved to {output_filename}")
