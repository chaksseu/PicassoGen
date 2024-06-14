import requests

def download_image(url, local_path):
        """
        Downloads an image from a URL and saves it to a local path.

        Args:
            url (str): The URL of the image to download.
            local_path (str): The local file path to save the downloaded image.
        """
        response = requests.get(url)
        
        if response.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"Image successfully downloaded: {local_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
