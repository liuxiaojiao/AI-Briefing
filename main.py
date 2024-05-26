from crewai import Crew
from textwrap import dedent
from ai_briefing_agents import AI_Briefing_Agents
from ai_briefing_tasks import AI_Briefing_Tasks
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

class AI_Briefing_Crew:

  def __init__(self, paper_category, top_voice_source):
    self.paper_category = paper_category
    self.top_voice_source = top_voice_source

  def run(self):
    agents = AI_Briefing_Agents()
    tasks = AI_Briefing_Tasks()

    innovation_researcher_agent = agents.innovation_researcher()
    paper_digestor_agent = agents.paper_digestor()
    top_voice_curator_agent = agents.top_voice_curator()
    content_strategist_agent = agents.content_strategist()

    ai_innovation_research_task = tasks.ai_innovation_research(
        innovation_researcher_agent
    )
    ai_paper_digestion_task = tasks.ai_paper_digestion(
        paper_digestor_agent,
        self.paper_category
    )
    ai_top_voice_curation_task = tasks.ai_top_voice_curation(
        top_voice_curator_agent,
        self.top_voice_source
    )
    ai_content_generation_task = tasks.ai_content_generation(
        content_strategist_agent
    )

    crew = Crew(
        agents=[
            paper_digestor_agent, innovation_researcher_agent, top_voice_curator_agent,  content_strategist_agent
            ],
        tasks=[
            ai_paper_digestion_task, ai_innovation_research_task, ai_top_voice_curation_task, ai_content_generation_task
            ],
        verbose=True,
        memory=True
    )

    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("## Welcome to AI Briefing Crew")
    print('-------------------------------')
    # paper_category = input(
    #     dedent("""
    #     What paper category do you want to scrape and analyze?
    #     """))
    
    # top_voice_source = input(
    #     dedent("""
    #     What website do you want to scrape and analyze for top voice?
    #     """))

    # ai_briefing_crew = AI_Briefing_Crew(paper_category, top_voice_source)
    # result = ai_briefing_crew.run()
    # print("\n\n########################")

    # print("## Here is your AI briefing letter:")
    # print("########################\n")
    # print(result)

    user_inputs = {
        'paper_category': 'cs.AI',
        'top_voice_source': 'deeplearning.ai'
    }

    result = AI_Briefing_Crew(paper_category = user_inputs['paper_category'], 
                          top_voice_source = user_inputs['top_voice_source']).run()

    print(result)
    # # Save the output to a .txt file
    dt_today = datetime.now().date()
    with open(f'output/AI_Briefing_{dt_today}.txt', 'w') as f:
        f.write(result)
