'''
Use arXiv API to scrape the latest paper abstracts published on arXiv - cs:AI 
Summarize the major findings.
'''

import pandas as pd
import arxiv
from datetime import datetime
from crewai import Agent, Task
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()
# Get the OPENAI_API_KEY
openai_api_key = os.getenv('OPENAI_API_KEY')
# Create the llm
llm_35_turbo = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo') # Loading GPT-3.5-turbo model


class PaperScraperDigestorTools():
    @tool('arXiv paper scraping and summarization')
    def arxiv_paper_search_digestion(category: str):
        """
        Scrape the abstracts from the latest 30 papers published in the arXiv - cs:AI category, 
        meticulously extracting pivotal findings and trends.
        """
        search = arxiv.Search(
            # scrape the latest papers published on arXiv - cs:AI
            query = category,
            # scrape the latest 30 papers
            max_results = 30,
            # sort the papers by the date they were submitted
            sort_by = arxiv.SortCriterion.SubmittedDate,
            sort_order = arxiv.SortOrder.Descending
        )

        paper_metadata = []
        for result in search.results():
            temp = ["","","","",""]
            temp[0] = result.title
            temp[1] = result.published
            temp[2] = result.entry_id
            temp[3] = result.summary
            temp[4] = result.pdf_url
            paper_metadata.append(temp)

        column_names = ['Title','Date','ID','Abstract','URL']
        df = pd.DataFrame(paper_metadata, columns=column_names)
        df['Date'] = df['Date'].apply(lambda x: x.date())
        dt_today = datetime.now().strftime("%Y%m%d")
        df.to_excel('output/final_scraped_arxiv_papers.xlsx', sheet_name=dt_today)

        print("Number of papers extracted : ", df.shape[0])

        important_findings = []
        for chunk in df['Abstract'].head(15):

            agent = Agent(
                    role='Principal Researcher',
                    goal=
                    'Do amazing researches and extract important findings or trends based on the content you are working with',
                    backstory=
                    "You're a Principal Researcher at a big company and your job is to translates complex research outcomes into clear, digestible insights.",
                    allow_delegation=False,
                    llm=llm_35_turbo
                    )

            task = Task(
                    agent=agent,
                    description=
                    f'Extract important findings and translate these complex research outcomes into clear, digestible insights from the content below.\n\nCONTENT\n-----\n{chunk}'
                )

            findings = task.execute()
            important_findings.append(findings)
        
        # print(important_findings)

        # Save the output to a .txt file
        # with open('output/important_findings_AI_paper.txt', 'w') as f:
        #     for finding in important_findings:
        #         f.write(finding + '\n')

        return "\n\n".join(important_findings)


   


