# Import das libs
import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# Criando Yahoo Finance Tool
def fetch_stock_price(ticket):
  stock = yf.download(ticket, start='2023-08-08', end='2024-08-20')
  return stock

yahoo_finance_tool = Tool(
  name = "Yahoo Finance Tool",
  description = "A tool to fetch stock price of {ticket} from Yahoo Finance",
  func = lambda ticket: fetch_stock_price(ticket)
)

# Importando OpenAI LLM - GPT
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent(
  role = "Senior Stock Price Analyst",
  goal = "To provide stock price analysis for a given stock ticket",
  backstory = "You're a highly experienced in analyzing the price of an specific stock and make predictions about its future price.",
  verbose = True,
  llm = llm,
  max_iter = 5,
  memory = True,
  tools = [yahoo_finance_tool],
  allow_delegation = False,
)

get_stock_price = Task(
  description = "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways.",
  expected_output = """ Specify the current trend stock price - up, down or sideways.
  eg. stock = 'APPL, price UP'
  """,
  agent = stockPriceAnalyst,
)

# Importando a Tool de search do DuckDuckGo
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

newsAnalyst = Agent(
  role = "Stock News Analyst",
  goal = "Create a short summary of the news related to a specific stock {ticket} company. Specify the current trend - up, down or sideways with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
  backstory = """ You're a highly experienced in analyzing the market trends and news and have tracked assest for more than 10 years.
  
  You're also master level analyst in the traditional markets and have a deep understanding of human psychology and market sentiment.
  
  You understand news, theirs tittles and information, but you look at those with a health dose of skepticism.
  You consider also the source of the news article.
  """,
  verbose = True,
  llm = llm,
  max_iter = 10,
  memory = True,
  tools = [search_tool],
  allow_delegation = False,
)

get_news = Task(
  description = f"""Take the stock and always include BTC to it (if not request).
  Use the search tool to search each one individually.
  
  The current date is {datetime.now()}
  
  Compose the results into a helpful report
  """,
  expected_output = """ A summary of the overeall market and one setence summary for each request asset.
  Include a fear/greed score for each asset based on the news. Use format:
  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
  agent = newsAnalyst,
)

stockAnalystWrite = Agent(
  role = "Senior Stock Analyst Writer",
  goal = """Analyze the trends price and news and write an insightful compelling and informative 3 paragraph long newsletter based on the stock report and price trend.
  """,
  backstory = """ You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
  
  You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything.
  """,
  verbose = True,
  llm = llm,
  max_iter = 10,
  memory = True,
  allow_delegation = True,
)

write_analyses = Task(
  description = """ Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
  that is brief and highlights the most important points.
  Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
  Include the previous analyses of stock trend and news summary.
  """,
  expected_output = """An eloquent 3 paragraph newsletter formated as markdown in an easy readable manner. It should contain:
  
  - 3 bullets executive summary
  - Introduction - set the overall picture and spike up the interest
  - main part provides the meat of the analysis including the news summary and fear/greed score
  - summary - key facts and concrete future trend predictions - up, down or sideways.
  """,
  agent = stockAnalystWrite,
  context = [get_stock_price, get_news],
)

crew = Crew(
  agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
  tasks = [get_stock_price, get_news, write_analyses],
  verbose = True,
  process= Process.hierarchical,
  full_output=True,
  share_crew=False,
  manager_llm=llm,
  max_iter=5,
)

# results = crew.kickoff(inputs={'ticket': 'AAPL'})

with st.sidebar:
  st.header('Enter the Stock to Research')
  
  with st.form(key='research_form'):
    topic = st.text_input(label='Select the ticket')
    submit_button = st.form_submit_button(label='Run Research')
    
if submit_button:
  if not topic:
    st.error('Please enter a stock ticket to research')
  else:
    results = crew.kickoff(inputs={'ticket': topic})
    
    st.subheader('Result of your research')
    st.write(results.raw)
