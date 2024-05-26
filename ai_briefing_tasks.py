from crewai import Task
from textwrap import dedent
from datetime import date


class AI_Briefing_Tasks():

  def ai_innovation_research(self, agent):
    return Task(description=dedent(f'''
        Conduct a comprehensive analysis of the latest advancements in AI in 2024 through online search. Identify key trends, breakthrough technologies, and potential industry impacts. 
        Your final answer MUST deliver a detailed report on AI emerging trends and innovations. 
        {self.__tip_section()}'''),
        expected_output=("Full analysis report in bullet points"),
        agent=agent)

  def ai_paper_digestion(self, agent, paper_category):
    return Task(description=dedent(f'''
        Scrape the abstracts from the latest papers published in the arXiv - cs:AI category, meticulously extracting pivotal findings and trends.
        Your final deliverable MUST be a distilled synthesis of contemporary AI research, presented in an easily understandable way.
        {self.__tip_section()}

        Paper Category: {paper_category}
        '''),
        expected_output=("Full analysis report in bullet points"),
        async_execution=True,
        agent=agent) 
  
  def ai_top_voice_curation(self, agent, top_voice_source):
    return Task(description=dedent(f'''
        Scrape the articles authored by leading voices from AI top voice source - deeplearning.ai, extracting key findings and trends related to AI development and its industrial applications.
        Your final deliverable MUST be a concise overview of the most important thoughts and developments in AI. 
        {self.__tip_section()}

        AI top voice source: {top_voice_source}'''),
        expected_output=("Full analysis report in bullet points"),
        async_execution=True,
        agent=agent) 

  def ai_content_generation(self, agent):
    return Task(description=dedent(f'''
        Using the insights provided from online search, paper AND AI top voices, develop an engaging blog post that highlights the most significant trends and advancements in AI development and industrial application. 
        Your post should be informative yet accessible, catering to a tech-savvy audience.
        Your final deliverable MUST clearly present the latest trends and advancements in AI development and industrial application. 
        {self.__tip_section()}'''),
        expected_output=("Detailed full blog post of at least 5 paragraphs"),
        agent=agent) 

  def __tip_section(self):
    return "If you do your BEST WORK, I'll tip you $100!"
