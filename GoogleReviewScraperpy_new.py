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
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from collections import defaultdict
import tqdm 
import keyboard


df = pd.read_excel('branch_locations_links_v2.xlsx')
 

x=0
y=766
user = ''
start=x
end=y
start_str = str(start)
end_str = str(end)
urls = df['url'][start:end]

data = defaultdict(list)

# driver_options = Options()
# driver_options.add_argument("--headless")

# chrome_options = Options()
# chrome_options.add_argument("--lang=de")
# driver = webdriver.Chrome(options=chrome_options)

profile_path = "C:/Users/KIIT/AppData/Local/Google/Chrome/User Data/Profile 1"  # Adjust as needed

options = webdriver.ChromeOptions()
options.add_argument(f"user-data-dir={profile_path}")

driver = webdriver.Chrome(options=options)
## FIRST RUN UNTIL HERE!!! ##
global stop_loop
for url in tqdm.tqdm(urls):
    driver.get(url)
    time.sleep(3)
    try:
        privacy_btn = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[1]/div/div/button')
        driver.execute_script("arguments[0].click();", privacy_btn)
        time.sleep(2)
    except:
        pass
    
# 1st Step click on the Weitere Rezensionen (45) button if it exists
    try:
        more_reviews_btn = driver.find_elements(By.XPATH,"//div[contains(@class, 'BgrMEd') and contains(@class, 'cYlvTc') and .//span[contains(text(), 'More reviews')]]")[-1]#[x for x in driver.find_elements(By.CLASS_NAME,'M77dve ') if 'Weitere Rezensionen' in x.text][0]
        driver.execute_script("arguments[0].click();", more_reviews_btn)
        # print("execute1")
    except IndexError:
        try:
            time.sleep(1)
            more_reviews_btn = driver.find_elements(By.XPATH,"//div[contains(@class, 'BgrMEd') and contains(@class, 'cYlvTc') and .//span[contains(text(), 'More reviews')]]")[-1]#[x for x in driver.find_elements(By.CLASS_NAME,'M77dve ') if 'Weitere Rezensionen' in x.text][0]
            driver.execute_script("arguments[0].click();", more_reviews_btn)
            print("execute2")
        except:
            try:
                time.sleep(1)
                more_reviews_btn = driver.find_elements(By.XPATH,"//div[contains(@class, 'BgrMEd') and contains(@class, 'cYlvTc') and .//span[contains(text(), 'More reviews')]]")[-1]#[x for x in driver.find_elements(By.CLASS_NAME,'M77dve ') if 'Weitere Rezensionen' in x.text][0]
                driver.execute_script("arguments[0].click();", more_reviews_btn)
                print("execute3")
            except IndexError:
                print("More Reviews not available!")
    try:
        sort_btn = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[7]/div[2]/button')
        driver.execute_script("arguments[0].click();", sort_btn)
        # print("sort found")
        time.sleep(2)
        newest_btn = driver.find_element(By.XPATH, '//*[@id="action-menu"]/div[2]/div/div')
        driver.execute_script("arguments[0].click();", newest_btn)    
        # print("newest found")
    except:
        try:
            sort_btn = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[7]/div[2]/button')
            driver.execute_script("arguments[0].click();", sort_btn)
            # print("sort found")
            time.sleep(2)
            newest_btn = driver.find_element(By.XPATH, '//*[@id="action-menu"]/div[2]/div/div')
            driver.execute_script("arguments[0].click();", newest_btn)
        except:
            print("not found")
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
                time.sleep(3)
                # selection_btn = driver.find_elements(By.XPATH,"//img[@class='eaLgGf']")[-1]
                selection_btn = driver.find_elements(By.XPATH,"//button[@class='PP3Y3d S1qRNe']")[-1]
                selection_btn.click()
            except:
                time.sleep(1)
                # selection_btn = driver.find_elements(By.XPATH,"//img[@class='eaLgGf']")[-1]
                # selection_btn = driver.find_elements(By.XPATH,"//button[@class='PP3Y3d S1qRNe']")[-1]
                pass

    time.sleep(5)
    actions = ActionChains(driver)
    actions.send_keys(Keys.ESCAPE)
    actions.perform()
    time.sleep(1)

    try:
        # num_reviews_text = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[3]').text
        # #//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[3]
        # # /html/body/div[2]/div[3]/div[8]/div[9]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[3]
        # print(num_reviews_text)
        num_reviews_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="QA0Szd"]/div/div/div[1]/div[2]/div/div[1]/div/div/div[2]/div[2]/div/div[2]/div[3]'))
        )
    
        # Get the text from the element
        num_reviews_text = num_reviews_element.text
        num_reviews = int(num_reviews_text.split(' ')[0].replace(',', ''))
        
    except Exception as e:
        print(f"Failed to load the number of reviews: {e}")
        num_reviews = 0
    # print(num_reviews)
    time.sleep(1)
    last_height =driver.execute_script("return document.body.scrollHeight")
    num_reviews_appeared = 0
    actions = ActionChains(driver)
    body = driver.find_element(By.XPATH, '//*[@id="QA0Szd"]/div/div')
    body.click()
    old_job=[]
    counter = 0
    while num_reviews_appeared < num_reviews:
        if keyboard.is_pressed('esc'):
            print("Esc key pressed. Exiting the loop.")
            break
        actions.send_keys(Keys.END)
        time.sleep(1)
        actions.perform()
        # print(f'num of reciews appeard = {num_reviews_appeared}')
        # break if reach 4 years, so we have 3 years of reviews
        num_reviews_appeared = len(driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']"))
        try:
            elements = driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']")
            if elements:
                last_obj=elements[-1]
                parsed_html = BeautifulSoup(last_obj.get_attribute('innerHTML'),"lxml")
                date = parsed_html.find_all(class_='rsqaWe')[0].text
                if date.split(' ')[1] == 'years' and int(date.split(' ')[0]) > 3:
                    # date == 'vor 4 Jahren':
                    break
        except:
            break
        # try:
        # elements = driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']")
        
        # if elements:
        #     counter+=1
        #     # print(elements)
        #     # time.sleep(1)
        #     last_obj=elements[-1]
        #     # if(old_job!=last_obj):
        #     parsed_html = BeautifulSoup(last_obj.get_attribute('innerHTML'),"lxml")
        #     date = parsed_html.find_all(class_='rsqaWe')[0].text
        #     # if(counter%10==0):
        #     #     old_job=last_obj
        #     if date.split(' ')[1] == 'years' and int(date.split(' ')[0]) > 3:
        #         # date == 'vor 4 Jahren':
        #         break
        #     else:
        #         # counter=0
        #         break
        # except Exception as e:
        #     print(e)
        #     break


# 3rd Step click in the "Mehr" button for the reviews with a lot of text - Each review belongs to the "MyEned" class (reviews = driver.find_elements(By.CLASS_NAME,'MyEned'))
    time.sleep(1)
    view_more_btn = driver.find_elements(By.XPATH,"//button[@class='w8nwRe kyuRq']")
    for y in view_more_btn:
        driver.execute_script("arguments[0].click();", y)


# 4th Step: extract all the relevant information from the reviews

    for w in driver.find_elements(By.XPATH,"//div[@class='jftiEf fontBodyMedium ']"): #Names 
        parsed_html = BeautifulSoup(w.get_attribute('innerHTML'),"lxml")
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
    # time.sleep(1)


df_reviews = pd.DataFrame(data)

df_reviews = df_reviews[df_reviews['Date'] != 'vor 4 Jahren']
df_reviews.rename(columns={'Url': 'url'}, inplace=True)

control_df = df_reviews.groupby('url')['Name'].count().reset_index().rename(columns={'Name':'Scraped Reviews'})
re_run_df = df.merge(control_df,how='left')



df_reviews.to_excel(f'Output_Files/reviews_{user}_{start}_to{end}.xlsx', index=False)

# original location table plus total number of reviews
re_run_df.dropna(subset='Scraped Reviews').to_excel(f'Processed_Files/branch_locations_links_{user}_{start}_to{end}.xlsx', index=False)


# # remedy if stopped
# urls = df['url'][:191]  # change to your range. This is for the case where the code stopped more than once
# last_url = df_reviews['url'].iloc[-1]
# index = urls.index[urls == last_url][0]
# # change the beginning number in line 20 into index+1, run line 20, then run from line 36

 