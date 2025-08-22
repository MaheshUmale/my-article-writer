import os
import json
import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser

# --- THE FIX IS HERE ---
# We must import BaseModel and Field directly from the main pydantic library
# for it to be compatible with CrewAI's Task validation.
from pydantic import BaseModel, Field
# --- END OF FIX ---

from typing import List

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Article Generation App",
    page_icon="✍️",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Helper Classes for Structured Output ---
# These classes now correctly inherit from the standard pydantic.BaseModel
class Image(BaseModel):
    url: str = Field(description="The direct URL of the image.")
    source: str = Field(description="The source webpage URL where the image was found.")

class ArticleOutput(BaseModel):
    article_text: str = Field(description="The final, fully formatted markdown article.")
    images: List[Image] = Field(description="A list of relevant images with their sources.")

# --- LLM and Tool Configuration ---
from crewai import Agent, LLM

llm = LLM(
    api_key=os.getenv("GEMINI_API_KEY"),
    model="gemini/gemini-1.5-flash",
)


search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

# --- Agent Definitions ---
strategist = Agent(
    role="C-Suite Content Strategist",
    goal="Reframe a technical topic into a compelling business narrative for a CXO audience.",
    backstory="A seasoned business consultant skilled in translating complex tech jargon into P&L-focused language.",
    llm=llm, verbose=True, allow_delegation=False
)
researcher = Agent(
    role="Expert Research Analyst",
    goal="Find compelling data, statistics, and relevant visual assets (charts, graphs, figures) to support a business argument.",
    backstory="A meticulous analyst with a knack for finding undeniable data and visually compelling graphics to make an argument powerful.",
    llm=llm, verbose=True, tools=[search_tool], allow_delegation=False
)
writer = Agent(
    role="Expert Business Ghostwriter",
    goal="Draft a clear, authoritative article based on a strategic outline and research findings.",
    backstory="A master wordsmith for technology executives, skilled in creating engaging narratives for a senior audience.",
    llm=llm, verbose=True, allow_delegation=False
)
csuite_reviewer = Agent(
    role="CIO Persona",
    goal="Review and refine an article draft to ensure it speaks directly to C-level concerns: profit, cost, and risk.",
    backstory="A pragmatic CIO who ruthlessly cuts fluff and ensures every sentence links technology to business value.",
    llm=llm, verbose=True, allow_delegation=False
)
json_formatter = Agent(
    role="Data Formatter",
    goal="Consolidate the final article text and visual research into a single, clean JSON object.",
    backstory="A meticulous data engineer who ensures outputs are perfectly structured and machine-readable.",
    llm=llm, verbose=True, allow_delegation=False
)

# --- Task Definitions ---
task_strategize = Task(
    description="1. Analyze the provided topic: '{topic}' and core insight: '{insight}'.\n2. Generate a compelling article title and a strategic angle for a CXO audience.",
    expected_output="A clear article title and a paragraph outlining the strategic angle.",
    agent=strategist
)
task_research = Task(
    description=(
        "Using the article angle, find supporting data and visuals.\n"
        "1. Find 5-7 compelling data points or statistics with source URLs.\n"
        "2. **Crucially, perform an image search to find 2-3 relevant charts, graphs, or figures.** For each image, you MUST provide both the direct image URL and the source webpage URL."
    ),
    expected_output=(
        "A research report formatted in markdown.\n"
        "The report must have two sections:\n"
        "### Textual Research\n- Data point 1 [Source URL]\n- Data point 2 [Source URL]\n\n"
        "### Visual Research\n- Image 1: [Direct Image URL] | Source: [Webpage URL]\n- Image 2: [Direct Image URL] | Source: [Webpage URL]"
    ),
    agent=researcher, context=[task_strategize]
)
task_draft = Task(
    description="Create a 900-word article draft using the provided angle and textual research. Do not include the images in this draft.",
    expected_output="A well-written, coherent first draft of the article.",
    agent=writer, context=[task_strategize, task_research]
)
task_review = Task(
    description="Review the article draft from a CIO's perspective, rephrasing technical jargon into clear business outcomes.",
    expected_output="A revised version of the article focused on business value.",
    agent=csuite_reviewer, context=[task_draft]
)
task_format_json = Task(
    description=(
        "Consolidate the final revised article and the visual research into a single JSON object. "
        "Use the research task's output to extract the image URLs and their sources."
    ),
    expected_output="A single, clean JSON object matching the Pydantic schema provided.",
    agent=json_formatter,
    context=[task_review, task_research],
    output_pydantic=ArticleOutput # This will now pass validation
)

# --- Crew Definition ---
article_crew = Crew(
    agents=[strategist, researcher, writer, csuite_reviewer, json_formatter],
    tasks=[task_strategize, task_research, task_draft, task_review, task_format_json],
    process=Process.sequential,
    verbose=True,
    max_rpm=25
)

# --- Streamlit App UI ---
st.title("✍️ AI Article Generation App")
st.markdown("This app uses a team of AI agents to research and write a C-suite-focused article on any topic, complete with supporting graphics.")

st.sidebar.header("Configuration")
st.sidebar.markdown("Enter your topic and unique insight below. The more specific your insight, the better the article.")

topic = st.sidebar.text_area(
    "Enter the main topic for the article:",
    height=100,
    placeholder="e.g., The business case for implementing a Data Mesh in a large financial institution."
)
insight = st.sidebar.text_area(
    "Enter your core insight or angle:",
    height=150,
    placeholder="e.g., Most companies treat Data Mesh as a technology project, but it is actually an organizational and cultural shift. The real ROI comes from empowering domain teams, not from the tech itself."
)

if st.sidebar.button("Generate Article"):
    if topic and insight:
        with st.spinner("Agents are collaborating... This may take a few minutes."):
            try:
                inputs = {'topic': topic, 'insight': insight}
                result_obj = article_crew.kickoff(inputs=inputs)
                
                st.success("Article generated successfully!")

                st.subheader("Generated Article")   
                st.markdown(result_obj.pydantic.article_text)

                st.subheader("Collected Graphics & Charts")
                if result_obj.images:
                    for img in result_obj.pydantic.images:
                        st.image(img.url, caption=f"Source: {img.source}")
                else:
                    st.write("No relevant images were found for this topic.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.sidebar.warning("Please provide both a topic and an insight.")

    