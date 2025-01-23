# scraping using beautiful soup 
from bs4 import BeautifulSoup
from dateutil import parser
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import time
import requests

# for coinness data scraping
def get_articles(headers, url):
    news_req = requests.get(url, headers=headers)
    soup = BeautifulSoup(news_req.content, "html.parser")
    # Extracting title
    title = soup.find("h1", {"class": "view_top_title noselect"})
    title_text = title.text.strip() if title else "No title found"
    # Finding the specific <div>
    article_content_div = soup.find('div', class_='article_content', itemprop='articleBody')
    content = ""  # Initialize content as empty string
    # Check if the div was found
    if article_content_div:
        # Extracting text from all <p> tags within the <div>
        p_tags = article_content_div.find_all('p')
        for p in p_tags:
            content += p.get_text(strip=True) + " "  # Appending each <p> content with a space for readability

        # Optionally, remove specific unwanted text
        unwanted_text = "뉴스 속보를 실시간으로...토큰포스트 텔레그램 가기"
        content = content.replace(unwanted_text, "").strip()
    else:
        content = "No content found in the specified structure."
    return title_text, content

def scrape_tokenpost():
    all_titles, all_contents, all_full_times = [], [], []
    for i in tqdm(range(1, 1201), desc="Scraping content from tokenpost"): # e.g. get pages 1 to 1201 example
        try:
            links = []
            url = f"https://www.tokenpost.kr/coinness?page={i}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            news_req = requests.get(url, headers=headers)
            soup = BeautifulSoup(news_req.content, "html.parser")
            elems = soup.find_all("div", {"class": "list_left_item"})
            for e in elems:
                article_elems = e.find_all("div", {"class": "list_item_text"})
                for article in article_elems:
                    title_link = article.find("a", href=True)
                    if title_link and '/article-' in title_link['href']:
                        full_link = 'https://www.tokenpost.kr' + title_link['href']
                        # Find the date element in the parent of the article
                        date_elem = article.parent.find("span", {"class": "day"})
                        if date_elem:
                            news_date = parser.parse(date_elem.text)
                            links.append(full_link)
                            all_full_times.append(news_date)
                        else:
                            # If no date is found, append a placeholder
                            all_full_times.append(np.nan)
            for link in links:
                try:
                    title, content = get_articles(headers, link)
                    all_titles.append(title)
                    all_contents.append(content)
                except Exception as e:
                    print(f"Error while scraping news content: {e}")
                    all_titles.append("Error fetching title")
                    all_contents.append("Error fetching content")
        except Exception as e:
            print(f"Error while scraping page {i}: {e}")
        time.sleep(0.1)
        
        # Save the DataFrame after every 200 pages
        if i % 200 == 0:
            # Ensure all lists have the same length by adding placeholders if needed
            max_len = max(len(all_titles), len(all_contents), len(all_full_times))
            all_titles.extend(["No title"] * (max_len - len(all_titles)))
            all_contents.extend(["No content"] * (max_len - len(all_contents)))
            all_full_times.extend([np.nan] * (max_len - len(all_full_times)))

            # Save a partial DataFrame
            temp_df = pd.DataFrame({
                'titles': all_titles,
                'contents': all_contents,
                'datetimes': all_full_times
            })
            temp_filename = f"tokenpost_partial_page_{i}.csv"
            temp_df.to_csv(temp_filename, index=False)
            print(f"Saved partial DataFrame: {temp_filename}")

    # Ensure all lists have the same length by adding placeholders if needed
    max_len = max(len(all_titles), len(all_contents), len(all_full_times))
    all_titles.extend(["No title"] * (max_len - len(all_titles)))
    all_contents.extend(["No content"] * (max_len - len(all_contents)))
    all_full_times.extend([np.nan] * (max_len - len(all_full_times)))

    return pd.DataFrame({'titles': all_titles, 'contents': all_contents, 'datetimes': all_full_times})

df = scrape_tokenpost()

df.to_csv("tokenpost_250123.csv", index=False) 

print(df)
