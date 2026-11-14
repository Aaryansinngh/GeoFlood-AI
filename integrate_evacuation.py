from pathlib import Path

path = Path("app.py")
content = path.read_text(encoding="utf-8")

# Add the UI import
import_line = "from src.evacuation.ui_tab import render_evacuation_tab"

if import_line not in content:
    commander_import = "from src.simulation.commander_view import render_commander_tab"
    if commander_import not in content:
        raise RuntimeError("Could not locate the Flood Commander import.")
    content = content.replace(
        commander_import,
        commander_import + "\n" + import_line,
        1
    )

# Add the ninth navigation label
research_label = '    "ðŸ”¬ Research & Validation",'
evacuation_label = '    "🚨 Evacuation Router",'

if evacuation_label not in content:
    if research_label not in content:
        raise RuntimeError("Could not locate the Research & Validation tab label.")
    content = content.replace(
        research_label,
        research_label + "\n" + evacuation_label,
        1
    )

# Add the ninth tab variable
old_assignment = """    tab_research,
) = st.tabs(TABS)"""

new_assignment = """    tab_research,
    tab_evacuation,
) = st.tabs(TABS)"""

if "tab_evacuation," not in content:
    if old_assignment not in content:
        raise RuntimeError("Could not locate the expected st.tabs assignment.")
    content = content.replace(old_assignment, new_assignment, 1)

# Add the tab renderer
render_marker = "# TAB: EVACUATION ROUTER"

if render_marker not in content:
    content += """

# ===========================================================================
# TAB: EVACUATION ROUTER
# ===========================================================================
with tab_evacuation:
    render_evacuation_tab()
"""

path.write_text(content, encoding="utf-8")
print("Evacuation Router integration added.")
