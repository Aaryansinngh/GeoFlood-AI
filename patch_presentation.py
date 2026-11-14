from pathlib import Path

p = Path("app.py")
s = p.read_text(encoding="utf-8")

# Replace the navigation definition with conditional navigation.
start = s.index("TABS = [")
end_marker = ") = st.tabs(TABS)"
end = s.index(end_marker, start) + len(end_marker)

new_navigation = '''if st.session_state["pres_mode"]:
    TABS = [
        "🏠 Overview",
        "🗺️ Flood Intelligence",
        "🚨 Evacuation Intelligence",
        "🔬 Research & Validation",
    ]
    tab_overview, tab_map, tab_evacuation, tab_research = st.tabs(TABS)
else:
    TABS = [
        "🌊 Overview",
        "🗺️ Flood Intel Map",
        "🛡️ Flood Commander",
        "🧪 Scenario Lab",
        "🛰️ Satellite Analysis",
        "⚖️ Model Intelligence",
        "📍 Location Explorer",
        "🔬 Research & Validation",
        "🚨 Evacuation Router",
    ]
    (
        tab_overview,
        tab_map,
        tab_commander,
        tab_scenario,
        tab_sat,
        tab_model,
        tab_location,
        tab_research,
        tab_evacuation,
    ) = st.tabs(TABS)
'''

s = s[:start] + new_navigation + s[end:]

# Hide developer sidebar in Presentation Mode.
sidebar_marker = "# PRESENTATION MODE BANNER"
if 'presentation-sidebar-style' not in s:
    idx = s.index(sidebar_marker)
    css = '''if st.session_state["pres_mode"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stHeader"] {background: transparent;}
        .block-container {
            max-width: 1450px;
            padding-top: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

'''
    s = s[:idx] + css + s[idx:]

# Add an exit button outside the hidden sidebar.
banner_line = 'if st.session_state["pres_mode"]:\n    PRES_STEPS = ['
replacement = '''if st.session_state["pres_mode"]:
    if st.button("✕ Exit Presentation Mode", key="exit_presentation_top"):
        st.session_state["pres_mode"] = False
        st.rerun()

    PRES_STEPS = ['''
assert banner_line in s
s = s.replace(banner_line, replacement, 1)

# Four presentation stages.
old_steps = '''    PRES_STEPS = [
        "Introduction",
        "Geographic Exploration",
        "Location Investigation",
        "Model Comparison",
        "Scenario Simulation",
        "Satellite Analysis",
        "Reliability & Limitations",
    ]'''
new_steps = '''    PRES_STEPS = [
        "Project Overview",
        "Flood Intelligence",
        "Evacuation Demonstration",
        "Research & Validation",
    ]'''
assert old_steps in s
s = s.replace(old_steps, new_steps, 1)
s = s.replace("Step {step}/7:", "Step {step}/4:", 1)
s = s.replace("step < 7", "step < 4", 1)

# Hide tabs that are not part of the presentation.
# Their complete existing code remains unchanged for Full Dashboard mode.
for token in [
    "with tab_commander:",
    "with tab_scenario:",
    "with tab_sat:",
    "with tab_model:",
    "with tab_location:",
]:
    pos = s.index(token)
    next_marker = s.find("\n# ===", pos)
    if next_marker == -1:
        raise RuntimeError(f"Could not find end of section: {token}")

    section = s[pos:next_marker]
    if not section.startswith("    "):
        indented = "\n".join(
            ("    " + line if line.strip() else line)
            for line in section.split("\n")
        )
        section = 'if not st.session_state["pres_mode"]:\n' + indented
        s = s[:pos] + section + s[next_marker:]

p.write_text(s, encoding="utf-8")
print("Presentation Mode patch applied.")
