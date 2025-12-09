# Web Scraping and AI-based Review Analysis

Hi! This is my project where I built a **complete pipeline to extract, process, and analyze customer reviews** from Google Maps. I wanted to create something that could **turn unstructured review text into actionable insights** using web scraping, NLP, and AI.

The goal of this project is to help businesses understand **customer sentiment**, identify **recurring issues**, and extract **key topics** from reviews automatically.

---

## About the Project

I realized that customer reviews contain a lot of valuable insights, but analyzing them manually is **tedious and error-prone**, especially for businesses with multiple locations and thousands of reviews.

So, I built a pipeline that automates this process end-to-end:

1. **Scraping Reviews:**
   I used **Selenium** and **BeautifulSoup** to scrape reviews from Google Maps. The scraper handles dynamic content, pop-ups, “load more” buttons, and scrolling automatically to capture all reviews.

2. **Translation & Cleaning:**
   Many reviews are in languages other than English, so I integrated **Google Translate** (and optionally DeepL) to translate the text. Then I clean the text by:

   * Removing stopwords, punctuation, and numbers
   * Lemmatizing words
   * Preparing the text for sentiment analysis and topic extraction

3. **Sentiment Analysis:**
   I implemented a **RoBERTa-based sentiment model** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to classify reviews as Positive, Neutral, or Negative.

   * I use a **custom scoring function** to convert raw model outputs into a bounded sentiment score (-1 to 1).
   * Long reviews are split into chunks, and scores are averaged for accurate sentiment.

4. **Topic Extraction:**
   Using **GPT models via Azure OpenAI and LangChain**, I extract the topics mentioned in reviews, like:

   * Customer Service
   * Store Location
   * Products
   * Store Atmosphere
   * Price
     I also capture the **specific sub-sentences** that mention each topic, which allows for more granular analysis.

5. **Data Consolidation & Export:**
   Finally, all the processed data is saved in structured Excel files including:

   * Original, translated, and cleaned review text
   * Sentiment scores and labels
   * Extracted topics
   * Location metadata

---

## Workflow Diagram

Here’s a visual overview of how the pipeline works:

```
Google Maps Reviews
         |
         v
   Web Scraping
         |
         v
Translation & Cleaning
         |
         v
   Sentiment Analysis
         |
         v
   GPT Topic Extraction
         |
         v
   Consolidated Output
     (Excel Files)
```

---

## Applications

I built this project to showcase how **AI and automation** can help businesses make sense of customer feedback. Some use cases include:

* **Retail & E-commerce Analytics:** Track store performance and satisfaction.
* **Market Research:** Detect recurring issues or popular products.
* **Customer Experience Management:** Quickly identify and act on service problems.
* **AI/NLP Research:** Demonstrates an end-to-end pipeline from scraping to topic-level insights.

---

## Advantages of My Approach

* **Fully Automated:** Once set up, it scrapes and analyzes reviews without manual intervention.
* **Multi-Language Support:** Handles non-English reviews with translation.
* **Granular Insights:** Captures topics and sub-sentences, not just sentiment.
* **Scalable:** Works on thousands of reviews across multiple locations.
* **Modular:** Each script can be reused or adapted for other projects.

---

This project was a fun and challenging way for me to combine **web scraping, NLP, sentiment analysis, and AI-based topic extraction** into a single workflow. I’m proud of how it can take messy, multilingual reviews and turn them into **actionable insights**.


