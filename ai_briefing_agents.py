from crewai import Agent
from langchain.llms import OpenAI
from tools.paper_scraper_digestor_tools import PaperScraperDigestorTools
from tools.top_voice_scraper_curator_tools import TopVoiceScraperCuratorTools
from tools.search_tools import SearchTools
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
# Create the llm
llm_35_turbo = ChatOpenAI(api_key=openai_api_key, model='gpt-3.5-turbo')


class AI_Briefing_Agents():

  def innovation_researcher(self):
    return Agent(
        role='AI Innovation Expert',
        goal='Uncover the latest cutting-edge developments in AI only through online search',
        backstory='Your expertise lies in searching online to discover cutting-edge developments in AI, \
            staying ahead of the curve by identifying emerging trends and innovations. \
            Your finding will serve as an essential resource for staying ahead in the rapidly evolving field of artificial intelligence',
        tools=[
            SearchTools.search_internet
        ],
        llm=llm_35_turbo,
        allow_delegation=False,
        verbose=True,
        max_iter=5)

  def paper_digestor(self):
    return Agent(
        role='AI Paper Digestor',
        goal='Extract important findings from latest published research papers',
        backstory="""You are responsible for scraping the latest paper abstracts published on arXiv - cs:AI \
            and extracting their most important findings. \
            Your finding translates these complex research outcomes into clear, digestible insights. \
            This process ensures that stakeholders receive succinct and accessible summaries, \
            enabling them to grasp the essence of cutting-edge AI developments swiftly""",
        tools=[
            PaperScraperDigestorTools.arxiv_paper_search_digestion
        ],
        llm=llm_35_turbo,
        allow_delegation=False,
        verbose=True,
        max_iter=5)

  def top_voice_curator(self):
    return Agent(
        role='AI Insights Curator',
        goal='Deliver human expert insights of AI development and application from AI top voices',
        backstory="""You are responsible for scraping the articles authored by leading voices from AI top voice source - deeplearning.ai \
            and distilling their most important insights. \
            Your findings facilitates informed decision-making by simplifying AI top voices into easily understood summaries.""",
        tools=[
            TopVoiceScraperCuratorTools.top_voice_curation,
        ],
        llm=llm_35_turbo,
        allow_delegation=False,
        verbose=True,
        max_iter=5)

  def content_strategist(self):
    return Agent(
        role='Tech Content Strategist',
        goal="""Craft compelling content on tech advancements""",
        backstory="""You are a renowned Content Strategist concentrating on AI development and industrial application. \
            Your expertise lies in crafting compelling content on tech advancements. \
            You will synthesize information from BOTH paper and AI top voice into engaging and informative narratives suitable for a broad audience.""",
        llm=llm_35_turbo,
        allow_delegation=True,
        verbose=True,
        max_iter=5)
