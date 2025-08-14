import os
import random
import json
import re
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool, WebsiteSearchTool, FirecrawlScrapeWebsiteTool
from docx import Document
from docx.shared import WD_ALIGN_PARAGRAPH
from PIL import Image, ImageDraw

from src.file_processor import FileProcessor
from src.persona import Persona

# Configure Gemini LLM
gemini_llm = LLM(
    model="gemini/gemini-2.0-flash-lite",
    temperature=0.7,
)

# ---------------- Helper Functions ----------------

def generate_avatar(name, color):
    """Generate a simple circular avatar with initials."""
    name = name or "Unknown"
    color = color or "#4CAF50"
    initials = "".join([n[0].upper() for n in name.split() if n])
    if not initials:
        initials = "?"
    initials = initials[:2]
    img = Image.new('RGB', (100, 100), color=color)
    d = ImageDraw.Draw(img)
    d.ellipse((5, 5, 95, 95), fill=color)
    d.text((50, 50), initials, fill="white", anchor="mm")
    return img

# ---------------- Create DOCX File ----------------
def markdown_to_word(markdown_content, topic):
    """Convert markdown content to Word document format."""
    doc = Document()

    # Add title
    title = doc.add_heading(f"Board Discussion: {topic}", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add date
    date_paragraph = doc.add_paragraph()
    date_run = date_paragraph.add_run(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    date_run.italic = True
    date_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Split the markdown content into sections
    sections = markdown_content.split("## ")

    # Process each section
    for i, section in enumerate(sections):
        if i == 0:  # Skip the title section as we've already added it
            continue

        # Extract section title and content
        lines = section.strip().split("\n")
        section_title = lines[0]
        section_content = "\n".join(lines[1:]).strip()

        # Add section heading
        doc.add_heading(section_title, level=2)

        # Process content based on section type
        if "Participants" in section_title:
            # Each participant is on a line starting with -
            for line in section_content.split("\n"):
                if line.startswith("- "):
                    # Remove the bullet and add as paragraph
                    participant_text = line[2:].strip()
                    p = doc.add_paragraph()
                    p.add_run("• ").bold = True
                    p.add_run(participant_text)

        elif "Research Findings" in section_title:
            # Add research findings as paragraphs
            for paragraph in section_content.split("\n\n"):
                if paragraph.strip():
                    doc.add_paragraph(paragraph.strip())

        elif "Discussion Transcript" in section_title:
            # Process each message in the transcript
            for message in section_content.split("\n\n"):
                if "**" in message and "**:" in message:
                    # Extract speaker and content
                    speaker_part = message.split("**:", 1)[0] + "**"
                    content_part = message.split("**:", 1)[1].strip()

                    # Add as formatted paragraph
                    p = doc.add_paragraph()
                    p.add_run(speaker_part.replace("**", "")).bold = True
                    p.add_run(": " + content_part)

                    # Add spacing after each message
                    doc.add_paragraph("")
                else:
                    # Just add as regular paragraph if not a message
                    if message.strip():
                        doc.add_paragraph(message.strip())

        elif "Key Points" in section_title:
            # Process bullet points
            for line in section_content.split("\n"):
                if line.startswith("- "):
                    # Add as bullet point
                    point_text = line[2:].strip()
                    p = doc.add_paragraph()
                    p.add_run("• ").bold = True
                    p.add_run(point_text)

    return doc

def generate_word_export(system):
    """Generate a Word document from the discussion data."""
    # First generate the markdown
    md = system.generate_markdown_export()

    # Convert markdown to Word document
    doc = markdown_to_word(md, system.discussion_topic)

    # Save to a BytesIO object
    docx_io = BytesIO()
    doc.save(docx_io)
    docx_io.seek(0)

    return docx_io

# ---------------- Board Discussion System ----------------

class BoardDiscussionSystem:
    """Agentic board discussion system with dynamic personas and chat-like interface."""

    def __init__(self):
        # Tools
        self.web_scraper = None  # Initialize as None

        # Initialize other tools that don't require API keys
        self.file_tool = FileReadTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-2.0-flash-lite",
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="gemini-embedding-exp-03-07",
                        task_type="retrieval_document",
                    ),
                ),
            )
        )
        self.web_tool = WebsiteSearchTool()

        # Base agent for research tasks
        self.research_agent = self._create_research_agent()

        # Dynamic attributes
        self.personas: List[Persona] = []
        self.persona_agents: List[Agent] = []
        self.user_persona: Optional[Persona] = None
        self.discussion_topic: str = ""
        self.materials: List[Dict] = []
        self.research_findings: str = ""
        self.discussion_history: List[Dict] = []
        self.discussion_dynamics: Dict[str, int] = {
            'pace': 50,         # 0 (detailed) to 100 (brief)
            'creativity': 50    # 0 (conventional) to 100 (highly creative)
        }

    def initialize_web_scraper(self, api_key):
        """Initialize the web scraper with the provided API key."""
        if api_key:
            try:
                self.web_scraper = FirecrawlScrapeWebsiteTool(
                    api_key=api_key,
                    page_options={"onlyMainContent": True}
                )
                return True
            except Exception as e:
                print(f"Error initializing web scraper: {e}")
                return False
        return False

    def _create_research_agent(self):
        tools = [self.file_tool, self.web_tool]
        if self.web_scraper:
            tools.append(self.web_scraper)

        return Agent(
            role="Research Analyst",
            goal="Extract and summarize key information from documents and online sources.",
            backstory="""You are a skilled research analyst who excels at analyzing complex information
            from various sources and summarizing key insights.""",
            verbose=True,
            tools=tools,
            llm=gemini_llm
        )

    def create_persona_agent(self, persona: Persona):
        return Agent(
            role=f"{persona.name} - Board Member",
            goal=f"Participate in a board discussion as {persona.name} with a unique perspective.",
            backstory=f"""You are {persona.name} with a background in {persona.background}.
            Your expertise is in {persona.expertise} and you hold the following viewpoints: {persona.viewpoints}.
            Your communication style is {persona.communication_style} with assertiveness {persona.assertiveness}/10
            and cooperation {persona.cooperation}/10.
            Discussion Dynamics - Pace: {self.discussion_dynamics['pace']}, Creativity: {self.discussion_dynamics['creativity']}.""",
            verbose=True,
            llm=gemini_llm
        )

    def add_persona(self, persona: Persona):
        self.personas.append(persona)
        agent = self.create_persona_agent(persona)
        self.persona_agents.append(agent)
        return agent

    def set_user_persona(self, persona: Persona):
        persona.is_user = True
        self.user_persona = persona
        self.add_persona(persona)

    def process_materials(self, files=None, urls=None):
        processed = []
        if files:
            for file in files:
                content = FileProcessor.process_file(file)
                processed.append({
                    "name": file.name,
                    "type": "file",
                    "content": content
                })
        if urls:
            for url in urls:
                content = FileProcessor.process_url(url)
                processed.append({
                    "name": url,
                    "type": "url",
                    "content": content
                })
        self.materials = processed
        return processed

    def auto_generate_personas(self, num_personas: int) -> List[Persona]:
        """Generate detailed personas with unique personalities based on materials."""
        if not self.materials:
            return []

        # Create sample content from materials
        sample_content = "\n\n".join([f"{mat['name']}:\n{mat['content'][:1000]}" for mat in self.materials[:3]])

        # Create a prompt for generating diverse personas
        prompt = f"""Based on the following materials about "{self.discussion_topic}",
        create {num_personas} diverse board member personas with distinct personalities, backgrounds, and viewpoints.

        For each persona, provide:
        1. A realistic full name
        2. Detailed professional background
        3. Specific area of expertise related to {self.discussion_topic}
        4. Unique viewpoints and perspectives they would bring to the discussion
        5. Communication style (Professional, Academic, Direct, Diplomatic, or Technical)
        6. Assertiveness level (1-10)
        7. Cooperation level (1-10)

        Ensure the personas have contrasting personalities and perspectives to create a dynamic discussion.

        MATERIALS SAMPLE:
        {sample_content}

        Format as a JSON array with fields: name, background, expertise, viewpoints, communication_style, assertiveness, cooperation"""

        task = Task(
            description=prompt,
            expected_output="JSON array of detailed persona descriptions",
            agent=self.research_agent
        )

        crew = Crew(
            agents=[self.research_agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )

        try:
            result = crew.kickoff()

            # Extract JSON from the result

            # Try to find JSON array in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', result.raw, re.DOTALL)

            if json_match:
                json_str = json_match.group(0)
                personas_data = json.loads(json_str)
            else:
                # If no JSON array found, attempt to parse the entire result
                personas_data = json.loads(result.raw)

            # Create personas from parsed data
            personas = []
            for data in personas_data:
                # Generate a vibrant color for the avatar
                color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

                persona = Persona(
                    name=data.get('name', f"Board Member {len(personas)+1}"),
                    background=data.get('background', "Business professional"),
                    expertise=data.get('expertise', f"Expert in {self.discussion_topic}"),
                    viewpoints=data.get('viewpoints', "Has balanced perspectives"),
                    communication_style=data.get('communication_style', "Professional"),
                    assertiveness=min(max(int(data.get('assertiveness', 5)), 1), 10),
                    cooperation=min(max(int(data.get('cooperation', 5)), 1), 10),
                    color=color
                )
                personas.append(persona)

            return personas

        except Exception as e:
            print(f"Error generating personas: {e}")
            # Fallback to basic personas with distinct traits
            return [
                Persona(
                    name=f"{random.choice(['Dr.', 'Prof.', 'Ms.', 'Mr.'])} {random.choice(['Smith', 'Johnson', 'Lee', 'Garcia', 'Chen', 'Kumar', 'Müller', 'Rodriguez'])} {random.choice(['A.', 'B.', 'C.', 'D.', 'E.'])}",
                    background=f"Professional with {random.randint(5, 30)} years of experience in {random.choice(['finance', 'technology', 'healthcare', 'education', 'consulting', 'manufacturing'])}",
                    expertise=f"Expert in {random.choice(['strategic planning', 'digital transformation', 'risk management', 'innovation', 'operations', 'market analysis'])} related to {self.discussion_topic}",
                    viewpoints=f"Believes that {random.choice(['innovation is key', 'careful analysis is essential', 'people come first', 'efficiency drives success', 'sustainability matters most', 'adaptability is crucial'])} when discussing {self.discussion_topic}",
                    communication_style=random.choice(["Professional", "Academic", "Direct", "Diplomatic", "Technical"]),
                    assertiveness=random.randint(3, 8),
                    cooperation=random.randint(3, 8),
                    color="#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                ) for i in range(num_personas)
            ]

    def research_task(self, topic, specific_questions=None):
        materials_text = "\n\n".join(
            [f"--- {mat['name']} ---\n{mat['content'][:1000]}..." for mat in self.materials]
        )
        questions = ""
        if specific_questions:
            questions = "\n".join([f"- {q}" for q in specific_questions.split("\n") if q.strip()])
            questions = f"\n\nSpecific Questions:\n{questions}"
        prompt = f"""Analyze the following materials about: {topic}.
{questions}

MATERIALS:
{materials_text}

Provide a comprehensive analysis covering key facts, perspectives, consensus, disagreements, and implications."""
        task = Task(
            description=prompt,
            expected_output="A detailed analysis report.",
            agent=self.research_agent
        )
        return task

    def create_discussion_task(self, persona_agent, topic, research_findings, other_personas, previous_contributions=None):
        others = "\n".join([f"- {p.name}: {p.background} (Expertise: {p.expertise})" for p in other_personas])
        prev = f"PREVIOUS CONTRIBUTIONS:\n{previous_contributions}" if previous_contributions else ""
        prompt = f"""Participate in a board discussion on: {topic}

RESEARCH FINDINGS:
{research_findings}

{prev}

OTHER BOARD MEMBERS:
{others}

Respond as your persona with a perspective influenced by your background and expertise.
Discussion Dynamics:
- Pace: {self.discussion_dynamics['pace']}
- Creativity: {self.discussion_dynamics['creativity']}

Keep your response concise but thorough (150-350 words)."""
        task = Task(
            description=prompt,
            expected_output="A thoughtful board discussion contribution.",
            agent=persona_agent
        )
        return task

    def handle_user_input(self, user_input, topic, research_findings, discussion_history):
        history_text = "\n\n".join([f"{entry['persona']}: {entry['content']}" for entry in discussion_history])
        if not self.user_persona:
            return "Error: User persona not set."
        # Format the user input using the research agent (or user agent)
        prompt = f"""The user ({self.user_persona.name}) has provided this input during a board discussion:
{user_input}

Format this input to match the user's persona characteristics and integrate it smoothly into the ongoing discussion.
User Persona:
- Name: {self.user_persona.name}
- Background: {self.user_persona.background}
- Expertise: {self.user_persona.expertise}
- Viewpoints: {self.user_persona.viewpoints}
- Communication Style: {self.user_persona.communication_style}

Discussion History:
{history_text}

Return only the formatted contribution."""
        task = Task(
            description=prompt,
            expected_output="Formatted user contribution.",
            agent=self.persona_agents[[p.id for p in self.personas].index(self.user_persona.id)]
        )
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        result = crew.kickoff()
        return result.raw

    def get_ai_responses(self, topic, research_findings, discussion_history, responding_ids: List[str]):
        import queue
        import threading

        responses_queue = queue.Queue()
        threads = []
        history_text = "\n\n".join([f"{entry['persona']}: {entry['content']}" for entry in discussion_history])

        def run_crew_task(persona, agent):
            prompt = f"""Continue the board discussion on: {topic}

RESEARCH FINDINGS:
{research_findings}

Discussion History:
{history_text}

Respond as {persona.name} based on your expertise and viewpoints.
Discussion Dynamics:
- Pace: {self.discussion_dynamics['pace']}
- Creativity: {self.discussion_dynamics['creativity']}

Keep your response concise but thorough (150-350 words)."""
            task = Task(
                description=prompt,
                expected_output=f"A thoughtful response from {persona.name}.",
                agent=agent
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                verbose=True,
                process=Process.sequential
            )
            result = crew.kickoff()
            responses_queue.put({
                "persona_id": persona.id,
                "persona_name": persona.name,
                "content": result.raw
            })

        for i, persona in enumerate(self.personas):
            if persona.id in responding_ids:
                agent = self.persona_agents[i]
                thread = threading.Thread(target=run_crew_task, args=(persona, agent))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        responses = []
        while not responses_queue.empty():
            responses.append(responses_queue.get())

        return responses

    def run_initial_discussion(self, topic, rounds=1):
        if self.materials:
            research_task = self.research_task(topic)
            crew = Crew(
                agents=[self.research_agent],
                tasks=[research_task],
                verbose=True,
                process=Process.sequential
            )
            research_result = crew.kickoff()
            self.research_findings = research_result.raw
        else:
            self.research_findings = "No research materials provided."
        for round_num in range(1, rounds + 1):
            for i, agent in enumerate(self.persona_agents):
                if self.personas[i].is_user:
                    continue
                others = [p for j, p in enumerate(self.personas) if j != i and not p.is_user]
                prev = "\n\n".join([f"{entry['persona']}: {entry['content']}" for entry in self.discussion_history])
                task = self.create_discussion_task(agent, topic, self.research_findings, others, previous_contributions=prev)
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    verbose=True,
                    process=Process.sequential
                )
                result = crew.kickoff()
                self.discussion_history.append({
                    "persona_id": self.personas[i].id,
                    "persona": self.personas[i].name,
                    "content": result.raw
                })
        return {
            "research": self.research_findings,
            "discussion": self.discussion_history
        }

    def extract_key_points(self):
        if len(self.discussion_history) < 3:
            return ["Discussion in progress..."]
        conversation = "\n\n".join([f"{entry['persona']}: {entry['content']}" for entry in self.discussion_history[-20:]])
        prompt = f"""Extract 3-5 key points from this board discussion on "{self.discussion_topic}".
Format each point as a concise bullet point.

Discussion transcript:
{conversation}

Key points (as bullet points):
"""
        task = Task(
            description=prompt,
            expected_output="Key points as bullet points.",
            agent=self.research_agent
        )
        crew = Crew(
            agents=[self.research_agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        result = crew.kickoff()
        points = result.raw.split("\n")
        return [p.strip().lstrip("•-*").strip() for p in points if p.strip()]

    def generate_markdown_export(self):
        md = f"# Board Discussion: {self.discussion_topic}\n\n"
        md += f"*Date: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        md += "## Participants\n\n"
        for p in self.personas:
            md += f"- **{p.name}**: {p.background}\n"
        md += "\n## Research Findings\n\n" + self.research_findings + "\n\n"
        md += "## Discussion Transcript\n\n"
        for entry in self.discussion_history:
            md += f"**{entry['persona']}**: {entry['content']}\n\n"
        key_points = self.extract_key_points()
        if key_points:
            md += "## Key Points\n\n"
            for point in key_points:
                md += f"- {point}\n"
        return md
