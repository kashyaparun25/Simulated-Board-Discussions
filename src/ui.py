import os
import streamlit as st
import time
import random
from datetime import datetime
import matplotlib.pyplot as plt

from src.discussion_system import BoardDiscussionSystem, generate_avatar, generate_word_export
from src.persona import Persona

def process_user_message(system: BoardDiscussionSystem, message: str):
    """
    Process the user's message, add it to the discussion,
    and trigger sequential AI responses one after another.
    """
    # 1. Format and add the user's message to the history
    with st.spinner("Formatting your message..."):
        formatted_message = system.handle_user_input(
            message,
            system.discussion_topic,
            system.research_findings,
            system.discussion_history
        )
        system.discussion_history.append({
            "persona_id": system.user_persona.id,
            "persona": system.user_persona.name,
            "content": formatted_message,
            "timestamp": datetime.now()
        })

    # 2. Determine which AI personas will respond (1-2 for a more natural flow)
    responding_personas = random.sample(
        [p for p in system.personas if not p.is_user],
        min(random.randint(1, 2), len(system.personas) - 1)
    )
    responding_ids = [p.id for p in responding_personas]

    # 3. Get AI responses sequentially (one after another)
    with st.spinner(f"Waiting for responses from {len(responding_personas)} agents..."):
        responses = system.get_ai_responses_sequential(
            system.discussion_topic,
            system.research_findings,
            system.discussion_history,
            responding_ids
        )

        # Add responses to history (they're already added in get_ai_responses_sequential)
        # We don't need to add them again here since they were added during processing

    # 4. Rerun the app to display all new messages
    st.rerun()

def create_streamlit_app():
    st.set_page_config(page_title="🎯 Board Discussion Simulator", layout="wide")

    # Initialize session state variables if they don't exist
    if 'system' not in st.session_state:
        st.session_state.system = BoardDiscussionSystem()
    if 'chat_turn' not in st.session_state:
        st.session_state.chat_turn = 0
    if 'user_joined' not in st.session_state:
        st.session_state.user_joined = False
    if 'message_delay' not in st.session_state:
        st.session_state.message_delay = 3
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0  # Default to first tab

    system: BoardDiscussionSystem = st.session_state.system

    # Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Configuration")

        # API Keys section
        with st.expander("🔑 API Keys"):
            firecrawl_key = st.text_input("Firecrawl API Key", value=os.getenv("FIRECRAWL_API_KEY"), type="password")
            gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY"), type="password")
            if st.button("💾 Save API Keys"):
                os.environ["FIRECRAWL_API_KEY"] = firecrawl_key
                os.environ["GEMINI_API_KEY"] = gemini_key

                # Initialize the web scraper with the new API key
                if firecrawl_key:
                    success = system.initialize_web_scraper(firecrawl_key)
                    if success:
                        st.success("✅ API keys saved and web scraper initialized!")
                    else:
                        st.warning("⚠️ API keys saved but web scraper initialization failed.")
                else:
                    st.success("✅ API keys saved!")

        # Discussion Settings
        st.header("🎛️ Discussion Settings")
        system.discussion_dynamics['pace'] = st.slider("⚡ Discussion Pace", 0, 100, 50)
        system.discussion_dynamics['creativity'] = st.slider("🎨 Creativity Level", 0, 100, 50)
        st.session_state.message_delay = st.slider("⏱️ Message Delay (seconds)", 1, 10, 3)

        # Analysis and Export
        st.header("📊 Discussion Analysis")
        if system.discussion_history:
            key_points = system.extract_key_points()
            with st.expander("🎯 Key Points", expanded=True):
                for point in key_points:
                    st.write(f"• {point}")

            # Participation Stats
            stats = {}
            for entry in system.discussion_history:
                stats[entry['persona']] = stats.get(entry['persona'], 0) + 1

            if stats:
                with st.expander("📈 Participation Stats", expanded=True):
                    fig, ax = plt.subplots(figsize=(8, 4))
                    names = list(stats.keys())
                    counts = list(stats.values())
                    ax.barh(names, counts)
                    ax.set_xlabel("Messages")
                    plt.tight_layout()
                    st.pyplot(fig)

            if st.button("📥 Export Discussion", use_container_width=True):
                md = system.generate_markdown_export()

                # Create export options
                export_cols = st.columns(2)
                with export_cols[0]:
                    st.download_button(
                        "📄 Download as Markdown",
                        data=md,
                        file_name=f"discussion_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with export_cols[1]:
                    # Generate Word document
                    word_bytes = generate_word_export(system)
                    st.download_button(
                        "📝 Download as Word Document",
                        data=word_bytes,
                        file_name=f"discussion_{datetime.now().strftime('%Y%m%d')}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )

        # Reset Session button at bottom of sidebar
        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Initialize a new system
            st.session_state.system = BoardDiscussionSystem()
            st.session_state.chat_turn = 0
            st.session_state.user_joined = False
            st.session_state.message_delay = 3
            st.session_state.active_tab = 0
            st.success("✨ Session reset successfully!")
            st.rerun()

    # Main content area
    st.title("🎯 Board Discussion Simulator")

    # Tab navigation
    tab_titles = ["🎯 Setup", "📚 Materials", "👥 Personas", "👤 Your Persona", "💬 Discussion"]
    current_tab = st.session_state.active_tab

    # Display horizontal tab buttons
    tabs = st.tabs(tab_titles)

    # Setup Tab
    with tabs[0]:
        st.header("🎯 Discussion Setup")
        topic = st.text_area("Discussion Topic", key="setup_topic",
                             value=system.discussion_topic or "",
                             placeholder="Enter the main topic for the board discussion...")
        specific_questions = st.text_area("❓ Specific Questions (Optional)",
                            help="Enter one question per line",
                            key="setup_questions",
                            placeholder="Question 1\nQuestion 2\n...")

        num_initial_rounds = st.slider("🔄 Initial Discussion Rounds", 1, 3, 1)

        if st.button("✅ Save Setup", use_container_width=True):
            if topic:
                system.discussion_topic = topic
                st.success("✨ Discussion topic saved!")
                time.sleep(1)
                # Auto navigate to Materials tab
                st.session_state.active_tab = 1
                st.rerun()
            else:
                st.error("❌ Please enter a discussion topic")

    # Materials Tab
    with tabs[1]:
        st.header("📚 Research Materials")
        st.info("📝 Upload materials to enable automatic persona generation")

        material_type = st.radio("📎 Source Type",
                               ["📄 Files", "🔗 URLs", "📑 Both"])

        uploaded_files = None
        urls = None

        if material_type in ["📄 Files", "📑 Both"]:
            uploaded_files = st.file_uploader(
                "Upload Files",
                accept_multiple_files=True,
                type=["pdf", "docx", "xlsx", "xls", "pptx", "ppt", "txt"],
                help="Upload research documents"
            )
            if uploaded_files:
                st.success(f"📎 {len(uploaded_files)} files uploaded")

        if material_type in ["🔗 URLs", "📑 Both"]:
            urls = st.text_area(
                "Enter URLs (one per line)",
                help="Add web pages for research",
                key="material_urls"
            )
            if urls:
                url_list = [u.strip() for u in urls.split("\n") if u.strip()]
                st.success(f"🔗 {len(url_list)} URLs added")

        if st.button("📥 Process Materials", use_container_width=True):
            url_list = [u.strip() for u in urls.split("\n") if u.strip()] if urls else None
            with st.spinner("🔄 Processing materials..."):
                system.process_materials(
                    files=uploaded_files if uploaded_files else None,
                    urls=url_list
                )
            st.success("✨ Materials processed successfully!")
            time.sleep(1)

            if system.materials:
                st.subheader("📑 Processed Materials")
                for mat in system.materials:
                    with st.expander(f"{'📄' if mat['type']=='file' else '🔗'} {mat['name']}"):
                       st.text_area(
                            "Preview",
                            mat['content'][:500] + "...",
                            height=100,
                            disabled=True,
                            key=f"preview_{mat['name']}"
                        )

                # Auto navigate to personas tab
                st.session_state.active_tab = 2
                st.rerun()

    # Personas Tab
    with tabs[2]:
        st.header("👥 Define Board Member Personas")

        # Check if materials are available
        if not system.materials:
            st.warning("⚠️ Please upload and process materials first to enable automatic persona generation")
            st.info("💡 You can still create personas manually")

        # Display current personas if any exist
        if system.personas:
            st.subheader("Current Personas")
            num_personas = len([p for p in system.personas if not p.is_user])
            if num_personas > 0:
                persona_cols = st.columns(min(4, num_personas))
                idx = 0
                for persona in system.personas:
                    if not persona.is_user:  # Don't show user persona here
                        with persona_cols[idx % len(persona_cols)]:
                            st.image(generate_avatar(persona.name, persona.color), width=80)
                            st.markdown(f"**{persona.name}**")
                            with st.expander("Details"):
                                st.write(f"🎓 **Background:** {persona.background}")
                                st.write(f"🔍 **Expertise:** {persona.expertise}")
                                st.write(f"💭 **Viewpoint:** {persona.viewpoints}")
                                st.write(f"🗣️ **Style:** {persona.communication_style}")
                                st.write(f"⚡ **Assertiveness:** {'▪️' * persona.assertiveness}")
                                st.write(f"🤝 **Cooperation:** {'▪️' * persona.cooperation}")
                        idx += 1
                st.markdown("---")

        creation_method = st.radio(
            "Choose Method",
            ["🤖 Auto Generate", "✍️ Create Manually"]
        )

        if creation_method == "🤖 Auto Generate":
            if not system.materials:
                st.error("❌ Upload and process materials first to use automatic generation")
            else:
                num_personas = st.slider("Number of Personas", 2, 6, 4)
                if st.button("🎲 Generate Personas", use_container_width=True):
                    with st.spinner("🔄 Generating personas based on materials..."):
                        generated = system.auto_generate_personas(num_personas)
                        if generated:
                            for persona in generated:
                                system.add_persona(persona)
                            st.success(f"✨ Generated {len(generated)} personas!")
                            time.sleep(1)
                            # Auto navigate to user persona tab
                            st.session_state.active_tab = 3
                            st.rerun()
                        else:
                            st.error("❌ Could not generate personas from materials")

        else:
            st.subheader("✍️ Add New Persona")
            # Use a form with submit button
            with st.form(key="manual_persona_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Name")
                    background = st.text_input("Background")
                    expertise = st.text_input("Expertise")
                with col2:
                    viewpoints = st.text_area("Key Viewpoints", key="persona_viewpoints")
                    communication_style = st.selectbox(
                        "Communication Style",
                        ["Professional", "Academic", "Direct", "Diplomatic", "Technical"]
                    )
                    assertiveness = st.slider("Assertiveness", 1, 10, 5)
                    cooperation = st.slider("Cooperation", 1, 10, 5)

                # Form submit button
                submit_button = st.form_submit_button("➕ Add Persona", use_container_width=True)

            if submit_button:
                if all([name, background, expertise, viewpoints]):
                    new_persona = Persona(
                        name=name,
                        background=background,
                        expertise=expertise,
                        viewpoints=viewpoints,
                        communication_style=communication_style,
                        assertiveness=assertiveness,
                        cooperation=cooperation,
                        color="#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                    )
                    system.add_persona(new_persona)
                    st.success(f"✨ Added persona: {name}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Please fill in all fields")

        # Navigation buttons
        st.markdown("---")
        if st.button("Next: Create Your Persona →", use_container_width=True):
            st.session_state.active_tab = 3
            st.rerun()

    # User Persona Tab
    with tabs[3]:
        st.header("👤 Create Your Persona")
        st.info("🎯 Create your persona to join the discussion")

        if system.user_persona:
            st.success(f"✨ Your persona: {system.user_persona.name}")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(generate_avatar(system.user_persona.name,
                                      system.user_persona.color),
                        width=150, caption="Your Avatar")
            with col2:
                st.write(f"**Name:** {system.user_persona.name}")
                st.write(f"**Background:** {system.user_persona.background}")
                st.write(f"**Expertise:** {system.user_persona.expertise}")
                st.write(f"**Viewpoints:** {system.user_persona.viewpoints}")
                st.write(f"**Style:** {system.user_persona.communication_style}")
                st.write(f"**Assertiveness:** {'🔵' * system.user_persona.assertiveness}")
                st.write(f"**Cooperation:** {'🟢' * system.user_persona.cooperation}")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset Persona", use_container_width=True):
                    for i, persona in enumerate(system.personas):
                        if persona.is_user:
                            system.personas.pop(i)
                            system.persona_agents.pop(i)
                            break
                    system.user_persona = None
                    st.success("✨ User persona reset")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("Next: Start Discussion →", use_container_width=True):
                    st.session_state.active_tab = 4
                    st.rerun()
        else:
            # User form with proper submit button
            with st.form(key="user_persona_form"):
                st.subheader("✍️ Create Your Profile")
                col1, col2 = st.columns(2)
                with col1:
                    user_name = st.text_input("Your Name")
                    user_background = st.text_input("Your Background")
                    user_expertise = st.text_input("Your Area of Expertise")
                with col2:
                    user_viewpoints = st.text_area("Your Key Viewpoints")
                    user_comm_style = st.selectbox(
                        "Your Communication Style",
                        ["Professional", "Academic", "Direct", "Diplomatic", "Technical"]
                    )
                    user_assertiveness = st.slider("Your Assertiveness", 1, 10, 5,
                                                 help="How strongly you express opinions")
                    user_cooperation = st.slider("Your Cooperation", 1, 10, 5,
                                               help="How collaborative you are")

                # Form submit button
                persona_submit = st.form_submit_button("✨ Create My Persona", use_container_width=True)

            if persona_submit:
                if all([user_name, user_background, user_expertise, user_viewpoints]):
                    user_persona = Persona(
                        name=user_name,
                        background=user_background,
                        expertise=user_expertise,
                        viewpoints=user_viewpoints,
                        communication_style=user_comm_style,
                        assertiveness=user_assertiveness,
                        cooperation=user_cooperation,
                        color="#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]),
                        is_user=True
                    )
                    system.set_user_persona(user_persona)
                    st.success(f"✨ User persona created: {user_name}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Please fill in all fields")

    # Discussion Tab
    with tabs[4]:
        st.header("💬 Board Discussion")

        # Check prerequisites
        if not system.discussion_topic:
            st.warning("⚠️ Please set up a discussion topic first")
        elif not system.materials:
            st.warning("⚠️ Please upload some research materials first")
        elif len(system.personas) < 2:
            st.warning("⚠️ Please create at least 2 personas first")
        else:
            # Display current personas
            st.subheader("🎭 Current Participants")
            num_personas = len(system.personas)
            if num_personas > 0:
                persona_cols = st.columns(min(4, num_personas))
                for i, persona in enumerate(system.personas):
                    with persona_cols[i % len(persona_cols)]:
                        st.image(generate_avatar(persona.name, persona.color), width=60)
                        st.markdown(f"**{persona.name}**" + (" 👤" if persona.is_user else ""))

            # Discussion Area
            st.markdown("---")

            if not st.session_state.get('discussion_started', False):
                st.info("🎯 Press 'Start Discussion' to begin the board meeting")
                # Make this button more prominent
                if st.button("🎬 Start Discussion", key="start_discussion", use_container_width=True):
                    with st.spinner("🔄 Starting discussion..."):
                        st.session_state.discussion_started = True
                        # Initialize discussion with first round
                        results = system.run_initial_discussion(
                            system.discussion_topic,
                            rounds=1
                        )
                    st.success("✨ Discussion started!")
                    st.rerun()
            else:
                # Show research findings at the top
                with st.expander("📚 Research Summary", expanded=False):
                    st.markdown(system.research_findings)

                # Message display area
                st.subheader("💬 Discussion Messages")

                # Create chat container
                chat_container = st.container(height=400, border=True)
                with chat_container:
                    # Display each message with avatar
                    for entry in system.discussion_history:
                        is_user = system.user_persona and (system.user_persona.name == entry['persona'])
                        persona_color = next((p.color for p in system.personas if p.name == entry['persona']), "#888888")

                        with st.chat_message(
                            "user" if is_user else "assistant",
                            avatar=generate_avatar(entry['persona'], persona_color)
                        ):
                            st.markdown(f"**{entry['persona']}** {'👤' if is_user else ''}")
                            st.markdown(entry['content'])
                            st.caption(f"⏰ {entry.get('timestamp', datetime.now()).strftime('%H:%M:%S')}")

                # Discussion control buttons
                if not st.session_state.get('discussion_concluded', False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💬 Continue Discussion", key="continue_discussion", use_container_width=True):
                            with st.spinner("🔄 Generating next discussion messages..."):
                                # Add a new round of AI responses sequentially
                                # Select 1-2 random personas to respond
                                responding_personas = random.sample(
                                    [p for p in system.personas if not p.is_user],
                                    min(random.randint(1, 2), len(system.personas) - 1)
                                )
                                responding_ids = [p.id for p in responding_personas]
                                
                                # Get responses sequentially
                                responses = system.get_ai_responses_sequential(
                                    system.discussion_topic,
                                    system.research_findings,
                                    system.discussion_history,
                                    responding_ids
                                )
                                
                                # Note: responses are already added to history in get_ai_responses_sequential
                            st.success("✨ New messages added to discussion!")
                            st.rerun()
                    with col2:
                        if st.button("🏁 Conclude Discussion", key="conclude_discussion", use_container_width=True):
                            with st.spinner("🔄 Concluding discussion..."):
                                # Generate conclusion
                                from crewai import Task, Crew
                                conclusion_task = Task(
                                    description=f'''As a facilitator, summarize the key points of this board discussion on "{system.discussion_topic}".
                                    Highlight areas of consensus, disagreement, and next steps.
                                    Keep it concise but comprehensive, around 200-300 words.''',
                                    expected_output="A balanced conclusion summarizing the discussion",
                                    agent=system.research_agent
                                )
                                crew = Crew(
                                    agents=[system.research_agent],
                                    tasks=[conclusion_task],
                                    verbose=True,
                                    process=Process.sequential
                                )
                                conclusion = crew.kickoff()

                                # Add conclusion to history
                                system.discussion_history.append({
                                    "persona_id": "facilitator",
                                    "persona": "Meeting Facilitator",
                                    "content": f"**DISCUSSION CONCLUSION**\n\n{conclusion.raw}",
                                    "timestamp": datetime.now()
                                })

                                # Mark discussion as concluded
                                st.session_state.discussion_concluded = True
                            st.success("✨ Discussion successfully concluded!")
                            st.rerun()
                else:
                    # Show message that discussion is concluded
                    st.success("🏁 This discussion has been concluded. You can export the results or start a new session.")
                    if st.button("📊 View Analysis", use_container_width=True):
                        # Auto-expand the key points section in the sidebar
                        st.session_state.show_analysis = True
                        # Change tab to analysis (optional)
                        st.rerun()

                # User participation section
                st.markdown("---")
                if system.user_persona:
                    if not st.session_state.get('user_joined', False):
                        # Join button
                        st.info("👤 Click below to join the discussion as your persona")
                        if st.button("👋 Join Discussion", key="join_discussion", use_container_width=True):
                            st.session_state.user_joined = True
                            st.success("✨ You've joined the discussion!")
                            st.rerun()
                    else:
                        # Chat input for user messages
                        st.markdown(f"**Chat as {system.user_persona.name}**")

                        # Use a form for user input
                        with st.form(key="chat_form"):
                            user_message = st.text_area("Your message:",
                                                    key="user_message_input",
                                                    height=100,
                                                    placeholder=f"Type your message as {system.user_persona.name}...")

                            col1, col2 = st.columns([3, 1])
                            with col1:
                                send_message = st.form_submit_button("Send Message", use_container_width=True)
                            with col2:
                                leave_discussion = st.form_submit_button("👋 Leave", use_container_width=True)

                        if send_message and user_message:
                            process_user_message(system, user_message)
                            st.rerun()

                        if leave_discussion:
                            st.session_state.user_joined = False
                            st.info("You've left the discussion.")
                            st.rerun()

                else:
                    st.warning("👤 You need to create your persona before joining the discussion")
                    if st.button("➡️ Create My Persona", key="goto_persona_tab", use_container_width=True):
                        st.session_state.active_tab = 3
                        st.rerun()
