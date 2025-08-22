from litellm import completion
import os
import os
from dotenv import load_dotenv
import os
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
import litellm
from crewai import Agent, Crew, Process, Task
#from crewai_tools import TavilySearchResults  # CHANGED: Imported Tavily instead of Serper
# from langchain_google_genai import ChatGoogleGenerativeAI # CHANGED: Imported Google's model

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


# Load environment variables from .env file
load_dotenv()
api_key=os.environ['GOOGLE_API_KEY']
response = completion(
    model='gemini/gemini-2.5-pro', 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)

import litellm



class LiteLLMAgent(Agent):
    def __init__(self, role, goal, backstory):
        super().__init__(role=role, goal=goal, backstory=backstory, allow_delegation=False)

    def execute_task(self, task, context=None, tools=None):        
        try:
            response = litellm.completion(
              model="gemini/gemini-2.5-pro",   
            #   provider="gemini",
              messages=[{"role": "user", "content": task.description}]
          )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

llm = LiteLLMAgent(role="user", goal="Linkedin Article for CXOs", backstory="You are Subject matter Expert in AI and Consultant to CXOs, CTOs..")

print(response)