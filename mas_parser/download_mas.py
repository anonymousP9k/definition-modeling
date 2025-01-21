"""Module to download MAS articles."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from tqdm import tqdm

from parse_mas_utils import parse_path, ROOT_DIR

CHARACTER_PAGES_URL = ('https://feb-web.ru/feb/common/tree.asp?/feb/mas/mas-abc&amp;encyc=1&amp;'
                       'vtsrch=on')

last_parsed_url_path = ROOT_DIR / "mas_parser" / "data" / "last_parsed_url.txt"


def load_headers() -> dict:
    """
    Load request headers from a JSON file.

    :return: A dictionary containing the headers.
    """
    with open(ROOT_DIR / "mas_parser"/ "headers.json", "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def get_character_pages(driver: webdriver.Chrome) -> list[str]:
    """
    Retrieve character page URLs from the main page.

    :param driver: Selenium WebDriver instance.
    :return: A list of character page URLs.
    """
    driver.get(CHARACTER_PAGES_URL)
    driver.implicitly_wait(2)

    all_links = driver.find_elements(By.TAG_NAME, 'a')

    target_links = []

    for link in all_links:
        href = link.get_attribute('href')
        if not href:
            continue
        match = re.match(r'.+feb/mas/mas-abc/(\d+(?:-\d+)?)\.htm\?cmd=2&istext=1', href)
        if href and match:
            target_links.append(
                f"https://feb-web.ru/feb/common/tree.asp?/feb/mas/mas-abc/{match.group(1)}"
                f".htm&encyc=1&vtsrch=on")

    return target_links


def extract_identifier(url: str) -> str:
    """
    Extract a unique identifier from the URL.

    :param url: The URL to extract the identifier from.
    :return: The identifier.
    """
    # Example extraction, adjust based on URL format
    match = re.search(r'/mas-abc/(\d+(?:-\d+)?)/ma(\d+)\.htm', url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return "unknown"


def save_html_content_jsonl(identifier: str, html_content: str, jsonl_file_path: Path) -> None:
    """
    Append the HTML content of a given URL to a JSON file with the identifier as key.

    :param identifier: The unique identifier of the URL.
    :param html_content: The HTML content to save.
    :param jsonl_file_path: The path to the JSON file to append the content to.
    """
    try:
        data = {"id": identifier, "html": html_content}

        with open(jsonl_file_path, "a", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
            file.write("\n")
    except Exception as e:
        print(f"Error saving HTML content to JSON: {e}")


def fetch_and_save_html_content(url: str, json_file_path: Path) -> None:
    """
    Fetch HTML content from a URL and save it to a JSON file using an identifier.

    :param url: The URL to fetch the HTML content from.
    :param json_file_path: The path to the JSON file to append the content to.
    """
    failed_attempts = 0
    identifier = extract_identifier(url)

    while failed_attempts < 5:
        try:
            response = requests.get(url, headers=load_headers(), timeout=30)
            if response.status_code == 200:
                html_content = response.text
                save_html_content_jsonl(identifier, html_content, json_file_path)
                if not last_parsed_url_path.parent.exists():
                    last_parsed_url_path.parent.mkdir(parents=True, exist_ok=True)
                with open(last_parsed_url_path, "w", encoding="utf-8") as last_url_file:
                    last_url_file.write(url)
            else:
                print(f"Failed to retrieve the web page. Status code: {response.status_code}")
                raise Exception("Failed to retrieve the web page.")
            break
        except Exception as e:
            print(f"Couldn't get {url}: {e}")
            failed_attempts += 1
            time.sleep(15)


def get_articles_pages(driver: webdriver.Chrome, character_urls: list[str]) -> list[str]:
    """
    Retrieve article page URLs from character pages.

    :param driver: Selenium WebDriver instance.
    :param character_urls: A list of character page URLs.
    :return: A list of article page URLs.
    """
    target_links = []

    for character_url in tqdm(character_urls):
        driver.get(character_url)
        driver.implicitly_wait(5)

        all_links = driver.find_elements(By.TAG_NAME, 'a')

        for link in all_links:
            href = link.get_attribute('href')
            if not href:
                continue
            match = re.match(
                r'.+feb/mas/mas-abc/(\d+(?:-\d+)?)/ma(\d+)\.htm\?cmd=2&istext=1', href)
            if href and match:
                article_url = (f"https://feb-web.ru/feb/mas/mas-abc/{match.group(1)}"
                               f"/ma{match.group(2)}.htm?cmd=p&istext=1")
                target_links.append(article_url)

    return target_links


def save_articles_urls(articles_urls: list[str]) -> None:
    """
    Save article URLs to a file.

    :param articles_urls: A list of article URLs to save.
    """
    with open(ROOT_DIR / "mas_parser" / "data" / "urls.txt", "w", encoding="utf-8") as urls_file:
        urls_file.write("\n".join(articles_urls))


def load_article_urls(continue_from_the_last_url: bool) -> list[str] | None:
    """
    Load article URLs from a file.

    :param continue_from_the_last_url: Whether to continue from the last URL.
    :return: A list of article URLs if the file exists, otherwise None.
    """
    urls_path = ROOT_DIR / "mas_parser" / "data" / "urls.txt"

    if not urls_path.exists():
        return None

    with open(urls_path, "r", encoding="utf-8") as urls_file:
        article_urls = urls_file.read().split("\n")

    if continue_from_the_last_url and last_parsed_url_path.exists():
        with open(last_parsed_url_path, "r", encoding="utf-8") as last_url_file:
            return article_urls[article_urls.index(last_url_file.read().strip()) + 1:]

    return article_urls


def main() -> None:
    """Download MAS articles."""
    parser = argparse.ArgumentParser(description="Download MAS articles.")
    parser.add_argument("--continue-from-the-last-url", action="store_true",
                        default=False,
                        help="Continue downloading from the last URL.")
    parser.add_argument("--output-path", type=str,
                        default="mas_parser/data/mas_articles.jsonl",
                        help="The path to save the HTML content to.")
    args = parser.parse_args()

    article_urls = load_article_urls(args.continue_from_the_last_url)

    if not article_urls:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)

        character_urls = get_character_pages(driver)

        print("Getting pages of articles...")
        article_urls = get_articles_pages(driver, character_urls)

        print("Saving article urls...")
        save_articles_urls(article_urls)

    output_path = parse_path(args.output_path)

    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    elif not args.continue_from_the_last_url:
        remove_the_file = input("The output file exists and continue_from_the_last_url is False. "
                                "Clear the file? y/n")
        if remove_the_file == "y":
            output_path.unlink(missing_ok=True)
        else:
            print("Exiting...")
            sys.exit(0)

    print("Downloading articles...")
    for article_url in tqdm(article_urls):
        fetch_and_save_html_content(article_url, output_path)


if __name__ == "__main__":
    main()
