"""Custom CSS styling module for the Streamlit app.

This module centralizes all UI styling into a single reusable CSS string.
It defines a dark theme with modern gradients, custom fonts, and consistent spacing.
Using a separate module makes it easy to maintain and update styles without cluttering the main app.
"""

import streamlit as st

# Complete CSS stylesheet for the app (dark theme with gradient backgrounds)
CUSTOM_CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

:root {
    --bg-1: #0b1220;
    --bg-2: #101827;
    --surface: rgba(17, 24, 39, 0.88);
    --surface-strong: #0f172a;
    --card: rgba(17, 24, 39, 0.72);
    --text: #e5e7eb;
    --muted: #9ca3af;
    --accent: #38bdf8;
    --accent-2: #22c55e;
    --warning: #f97316;
    --border: rgba(148, 163, 184, 0.22);
    --shadow: 0 12px 30px rgba(2, 6, 23, 0.35);
}

.stApp {
    background:
        radial-gradient(800px 400px at 8% -10%, rgba(56, 189, 248, 0.12), transparent 60%),
        radial-gradient(900px 600px at 100% 10%, rgba(34, 197, 94, 0.10), transparent 60%),
        linear-gradient(180deg, var(--bg-1), var(--bg-2));
    color: var(--text);
}

html, body, [class*="css"] {
    font-family: "Source Sans 3", "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
    letter-spacing: 0.2px;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, rgba(56, 189, 248, 0.12), rgba(34, 197, 94, 0.08));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.75rem;
}

.hero-badge {
    display: inline-block;
    text-transform: uppercase;
    font-size: 0.72rem;
    letter-spacing: 0.2rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
}

.hero-subtitle {
    margin-top: 0.35rem;
    font-size: 1rem;
    color: var(--muted);
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 0.5rem 0;
}

.section-subtitle {
    color: var(--muted);
    margin-bottom: 1rem;
}

.divider {
    height: 1px;
    background: var(--border);
    margin: 1.25rem 0 1.5rem 0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.96));
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

div[data-testid="stFileUploader"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.75rem;
    box-shadow: var(--shadow);
}

.stImage img {
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow);
}

div[data-testid="stMetric"] label {
    color: var(--muted);
    font-weight: 600;
    letter-spacing: 0.3px;
}

div[data-testid="stMetricValue"] {
    color: var(--text);
}

div[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

div[data-testid="stExpander"] details {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
}

div[data-testid="stExpander"] summary {
    color: var(--text);
    font-weight: 600;
}

div[data-testid="stProgress"] > div {
    background-color: rgba(148, 163, 184, 0.2);
}

div[data-testid="stProgress"] > div > div {
    background-color: var(--accent);
}

#MainMenu {visibility: hidden;}
header[data-testid="stHeader"] {visibility: hidden;}
div[data-testid="stToolbar"] {visibility: hidden;}
footer {visibility: hidden;}
</style>
'''


def inject_custom_css():
    """Inject the shared custom CSS into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
