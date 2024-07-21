import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from collections import defaultdict
import tqdm 


df = pd.read_excel('branch_locations_links_v2.xlsx')


# user = 'xiao'
# urls = df['url'][:191]

# user = 'winnetou'
# urls = df['url'][190:382]

# user = 'ivan'
# urls = df['url'][381:573]

user = 'vindhya'
urls = df['url'][572:]

data = defaultdict(list)

driver = webdriver.Chrome()
## FIRST RUN UNTIL HERE!!! ##

for url in tqdm.tqdm(urls):
    driver.get(url)
    time.sleep(3)
    try:
        privacy_btn = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[1]/div/div/button')
        driver.execute_script("arguments[0].click();", privacy_btn)
        time.sleep(3)
    except:
        pass
    
# 1st Step click on the Weitere Rezensionen (45) button if it exists
    try:
        more_reviews_btn = driver.find_elements(By.XPATH,"//div[contains(concat(' ',@class,' '), ' BgrMEd cYlvTc ') and contains(., 'Weitere Rezensionen')]")[-1]#[x for x in driver.find_elements(By.CLASS_NAME,'M77dve ') if 'Weitere Rezensionen' in x.text][0]
        driver.execute_script("arguments[0].click();", more_reviews_btn)
    except IndexError:
        try:
            time.sleep(1)
            more_reviews_btn = driver.find_elements(By.XPATH,"//div[contains(concat(' ',@class,' '), ' BgrMEd cYlvTc ') and contains(., 'Weitere Rezensionen')]")[-1]#[x for x in driver.find_elements(By.CLASS_NAME,'M77dve ') if 'Weitere Rezensionen' in x.text][0]
            driver.execute_script("arguments[0].click();", more_reviews_btn)
        except:
            try:
                time.sleep(1)
                more_reviews_btn = driver.find_elements(By.XPATH,"//div[contains(concat(' ',@class,' '), ' BgrMEd cYlvTc ') and contains(., 'Weitere Rezensionen')]")[-1]#[x for x in driver.find_elements(By.CLASS_NAME,'M77dve ') if 'Weitere Rezensionen' in x.text][0]
                driver.execute_script("arguments[0].click();", more_reviews_btn)
            except IndexError:
                print("More Reviews not available!")
    try:
        sort_btn = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[8]/div[2]/button')
        driver.execute_script("arguments[0].click();", sort_btn)
        newest_btn = driver.find_element(By.XPATH, '//*[@id="action-menu"]/div[2]/div/div')
        driver.execute_script("arguments[0].click();", newest_btn)
    except:
        pass

   
# 2nd Step scroll down in the review section for 1 minute
    try:
        # selection_btn = driver.find_elements(By.XPATH,"//img[@class='eaLgGf']")[-1]
        selection_btn = driver.find_elements(By.XPATH,"//button[@class='PP3Y3d S1qRNe']")[-1]
        selection_btn.click()
    except:
        try:
            time.sleep(3)
            # selection_btn = driver.find_elements(By.XPATH,"//img[@class='eaLgGf']")[-1]
            selection_btn = driver.find_elements(By.XPATH,"//button[@class='PP3Y3d S1qRNe']")[-1]
            selection_btn.click()
        except:
            try:
                time.sleep(1)
                # selection_btn = driver.find_elements(By.XPATH,"//img[@class='eaLgGf']")[-1]
                selection_btn = driver.find_elements(By.XPATH,"//button[@class='PP3Y3d S1qRNe']")[-1]
                selection_btn.click()
            except:
                time.sleep(1)
                # selection_btn = driver.find_elements(By.XPATH,"//img[@class='eaLgGf']")[-1]
                selection_btn = driver.find_elements(By.XPATH,"//button[@class='PP3Y3d S1qRNe']")[-1]

    time.sleep(2)
    actions = ActionChains(driver)
    actions.send_keys(Keys.ESCAPE)
    actions.perform()
    time.sleep(1)

    try:
        num_reviews_text = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[1]/div/div[2]/div[3]').text
        num_reviews = int(num_reviews_text.split(' ')[0].replace('.', ''))
    except:
        num_reviews = 0

    time.sleep(3)

    num_reviews_appeared = 0
    while num_reviews_appeared < num_reviews:
        actions = ActionChains(driver)
        actions.send_keys(Keys.END)
        time.sleep(0.5)
        actions.perform()

        num_reviews_appeared = len(driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']"))
        
        # break if reach 4 years, so we have 3 years of reviews
        last_obj = driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']")[-1]
        parsed_html = BeautifulSoup(last_obj.get_attribute('innerHTML'))
        date = parsed_html.find_all(class_='rsqaWe')[0].text
        if date == 'vor 4 Jahren':
            break


# 3rd Step click in the "Mehr" button for the reviews with a lot of text - Each review belongs to the "MyEned" class (reviews = driver.find_elements(By.CLASS_NAME,'MyEned'))
    time.sleep(1)
    view_more_btn = driver.find_elements(By.XPATH,"//button[@class='w8nwRe kyuRq']")
    for y in view_more_btn:
        driver.execute_script("arguments[0].click();", y)


# 4th Step: extract all the relevant information from the reviews

    for w in driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']"): #Names 
        parsed_html = BeautifulSoup(w.get_attribute('innerHTML'))
        try: #Name
            data['Name'].append(parsed_html.find_all(class_='d4r55')[0].text)
        except:
            data['Name'].append("Name not found")
        try: #Info
            data['Info'].append(parsed_html.find_all(class_='WNxzHc qLhwHc')[0].text)
        except:
            data['Info'].append("Info not found")
        try: #Text
            data['Text'].append(parsed_html.find_all(class_='wiI7pd')[0].text)
        except:
            data['Text'].append("Text not found")
        try: #Star rating
            data['Rating'].append(parsed_html.find_all(class_='kvMYJc')[0]['aria-label'])
        except:
            data['Rating'].append("Rating not found")
        try: #Date
            data['Date'].append(parsed_html.find_all(class_='rsqaWe')[0].text)
        except:
            data['Date'].append("Date not found")
        data['url'].append(url)
        
    print(len(driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']")),"reviews found")


df_reviews = pd.DataFrame(data)
df_reviews = df_reviews[df_reviews['Date'] != 'vor 4 Jahren']
df_reviews.rename(columns={'Url': 'url'}, inplace=True)

control_df = df_reviews.groupby('url')['Name'].count().reset_index().rename(columns={'Name':'Scraped Reviews'})
re_run_df = df.merge(control_df,how='left')



df_reviews.to_excel(f'WEB_SCRAP/Output_Files/reviews_{user}.xlsx', index=False)

# original location table plus total number of reviews
re_run_df.dropna(subset='Scraped Reviews').to_excel(f'WEB_SCRAP/Processed_Files/branch_locations_links_{user}.xlsx', index=False)


