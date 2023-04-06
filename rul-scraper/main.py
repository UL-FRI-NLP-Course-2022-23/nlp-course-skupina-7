import json
import time

import requests as requests
from bs4 import BeautifulSoup

TOTAL_TO_SCRAPE = 5000


def main():
    url_prefix = "https://repozitorij.uni-lj.si/IzpisGradiva.php?id="

    final_collection = []
    successful = 0
    done = 0
    for diploma_id in range(145102, 145102 - TOTAL_TO_SCRAPE, -1):
        # Print the progress
        print(f"\rScraping {diploma_id} ({successful}/{done})", end="")
        time.sleep(0.5)
        done += 1

        url = f"{url_prefix}{diploma_id}"
        # Get the html content of the page
        page = requests.get(url)
        # Parse the html content
        soup = BeautifulSoup(page.content, 'html.parser')
        # Get all elements with the class "izpisPovzetka_omejeno"
        elements = soup.find_all(class_="izpisPovzetka_omejeno")
        # If there are no elements with the class "izpisPovzetka_omejeno" then the diploma does not exist
        if len(elements) == 0:
            continue

        # For each element with the class "izpisPovzetka_omejeno" get the text
        jeziki = {"id": diploma_id}
        for element in elements:
            text = element.get_text()
            # Figure out the language of the text
            if text.count("the") > 3:
                jeziki["eng"] = text
            else:
                jeziki["slv"] = text

        if "eng" in jeziki and "slv" in jeziki:
            final_collection.append(jeziki)
            successful += 1

    with open("output.json", "w") as f:
        f.write(json.dumps(final_collection))


if __name__ == '__main__':
    main()
