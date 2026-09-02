"""Sistema visual monocromático y compacto para la interfaz Streamlit.

Este módulo solo inyecta presentación. No consulta datos, no modifica estados
ni participa en cálculos financieros.
"""

from __future__ import annotations

import streamlit as st


PREMIUM_CSS = r"""
<style>
:root {
  --ui-black: #000000;
  --ui-bg: #080808;
  --ui-panel: #111111;
  --ui-panel-raised: #151515;
  --ui-border: #292929;
  --ui-border-strong: #3a3a3a;
  --ui-text: #f7f7f7;
  --ui-muted: #909090;
  --ui-faint: #666666;
  --ui-radius: 8px;
  --ui-gap: 12px;
}

html { scroll-behavior: smooth; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
  background: var(--ui-bg) !important;
}

[data-testid="stHeader"] {
  height: 2.75rem;
  background: rgba(0, 0, 0, .96) !important;
  border-bottom: 1px solid #1d1d1d;
}

.stMainBlockContainer,
[data-testid="stMainBlockContainer"] {
  box-sizing: border-box;
  width: 100%;
  max-width: 1580px;
  padding: 4.35rem clamp(1.1rem, 2.7vw, 2.75rem) 4rem !important;
  margin-inline: auto;
  animation: ui-enter 140ms ease-out both;
}

.stMainBlockContainer > [data-testid="stVerticalBlock"],
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
  gap: 1rem;
}

@keyframes ui-enter {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* Encabezado sobrio: ocupa su propia fila y nunca queda bajo la barra superior. */
.st-key-page_header {
  position: relative;
  box-sizing: border-box;
  width: 100%;
  padding: .2rem 0 1rem;
  margin: 0 0 .35rem;
  border: 0;
  border-bottom: 1px solid var(--ui-border);
  border-radius: 0;
  background: transparent;
}

.st-key-page_header h1 {
  margin: .2rem 0 .3rem;
  color: var(--ui-text);
  font-size: clamp(1.7rem, 3vw, 2.25rem);
  font-weight: 650;
  line-height: 1.13;
  letter-spacing: -.035em;
  text-wrap: balance;
}

.st-key-page_header [data-testid="stCaptionContainer"] {
  max-width: 74rem;
  color: var(--ui-muted);
  font-size: .84rem;
  line-height: 1.5;
}

.st-key-page_header [data-testid="stMarkdownContainer"] p:first-child {
  margin: 0 0 .15rem;
  color: #a0a0a0;
  font-size: .63rem;
  font-weight: 600;
  letter-spacing: .13em;
  text-transform: uppercase;
}

/* Navegación lateral monocromática. */
section[data-testid="stSidebar"] {
  background: #050505 !important;
  border-right: 1px solid #202020;
  box-shadow: none !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
  min-height: 2.35rem;
  margin: .1rem .55rem;
  padding-block: .42rem;
  color: #8d8d8d;
  border: 0;
  border-left: 2px solid transparent;
  border-radius: 4px;
  background: transparent;
  box-shadow: none;
  transition: color 120ms ease, background-color 120ms ease,
              border-color 120ms ease;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
  color: #d9d9d9;
  background: #0d0d0d;
}

section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
  color: #ffffff;
  border-left-color: #dedede;
  background: #111111;
  box-shadow: none;
}

.st-key-sidebar_market_status {
  margin: .7rem .55rem .25rem;
  padding: .7rem;
  border: 1px solid #242424;
  border-radius: var(--ui-radius);
  background: #0b0b0b;
}

/* Tarjetas y formularios compactos. */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"] {
  box-sizing: border-box;
  border: 1px solid var(--ui-border) !important;
  border-radius: var(--ui-radius) !important;
  background: var(--ui-panel) !important;
  box-shadow: none !important;
  transition: border-color 120ms ease, background-color 120ms ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stForm"]:hover {
  border-color: var(--ui-border-strong) !important;
  background: #121212 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
  gap: .7rem;
}

[data-testid="stHorizontalBlock"] {
  align-items: stretch;
  gap: var(--ui-gap);
}

/* Las filas de KPIs distribuyen exactamente el mismo ancho disponible. */
[data-testid="stHorizontalBlock"]:has(> div [data-testid="stMetric"]) > div {
  flex: 1 1 0 !important;
  min-width: 0 !important;
}

div[data-testid="stMetric"] {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  min-height: 6.1rem;
  padding: .72rem .82rem;
  overflow: hidden;
  border: 1px solid var(--ui-border) !important;
  border-radius: var(--ui-radius) !important;
  background: var(--ui-panel) !important;
  box-shadow: none !important;
  transition: border-color 120ms ease, background-color 120ms ease;
}

div[data-testid="stMetric"]:hover {
  border-color: var(--ui-border-strong) !important;
  background: var(--ui-panel-raised) !important;
}

div[data-testid="stMetricLabel"] {
  color: #a5a5a5;
  font-size: .71rem;
  font-weight: 550;
  line-height: 1.3;
  letter-spacing: .005em;
}

div[data-testid="stMetricValue"] {
  color: var(--ui-text);
  font-size: clamp(1.38rem, 2vw, 1.85rem);
  line-height: 1.2;
  letter-spacing: -.035em;
  font-variant-numeric: tabular-nums;
}

div[data-testid="stMetricDelta"] { font-size: .68rem; }

/* Botones planos, sin halo ni movimiento. */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
  min-height: 2.35rem;
  padding: .42rem .78rem;
  color: #dedede;
  border: 1px solid #303030;
  border-radius: 6px;
  background: #151515;
  box-shadow: none !important;
  transition: color 110ms ease, border-color 110ms ease,
              background-color 110ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
  color: #ffffff;
  border-color: #505050;
  background: #1d1d1d;
  box-shadow: none !important;
}

.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] > button:active { background: #242424; }

button[kind="primary"],
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
  color: #000000;
  border-color: #f0f0f0;
  background: #f0f0f0;
  box-shadow: none !important;
}

button[kind="primary"]:hover,
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
  color: #000000;
  border-color: #ffffff;
  background: #ffffff;
}

/* Pestañas tipo Linear: texto y línea inferior, sin cápsulas. */
[data-baseweb="tab-list"],
[role="tablist"] {
  gap: 1.2rem;
  min-height: 2.65rem;
  padding: 0;
  overflow-x: auto;
  border: 0;
  border-bottom: 1px solid var(--ui-border);
  border-radius: 0;
  background: transparent;
}

[data-baseweb="tab"],
div[data-testid="stTab"] {
  min-height: 2.65rem;
  padding-inline: .1rem;
  color: #777777;
  border: 0;
  border-bottom: 1px solid transparent;
  border-radius: 0;
  background: transparent !important;
  box-shadow: none !important;
  transition: color 110ms ease;
}

[data-baseweb="tab"]:hover,
div[data-testid="stTab"]:hover { color: #cfcfcf; }
[data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTab"][aria-selected="true"] {
  color: #ffffff;
  border-bottom-color: #ffffff;
  box-shadow: none !important;
}

div[data-testid="stTab"] .react-aria-SelectionIndicator {
  height: 1px !important;
  background: #ffffff !important;
}

[data-testid="stTabsScrollLeft"],
[data-testid="stTabsScrollRight"] {
  background: var(--ui-bg) !important;
  background-image: none !important;
  box-shadow: none !important;
}

[data-testid="stSegmentedControl"] > div {
  padding: .2rem;
  border: 1px solid #2b2b2b;
  border-radius: 6px;
  background: #0d0d0d;
}

input, textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
  border-radius: 6px !important;
  background-color: #0d0d0d !important;
  box-shadow: none !important;
}

input:focus, textarea:focus,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
  border-color: #696969 !important;
  box-shadow: 0 0 0 1px #696969 !important;
}

/* Tablas, gráficas, archivos y mensajes. */
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
  overflow: hidden;
  border-radius: 6px;
}

[data-testid="stPlotlyChart"] { border-radius: 0; }

[data-testid="stElementToolbarButtonContainer"] {
  border: 1px solid #292929;
  background: #111111 !important;
  box-shadow: none !important;
}

section[data-testid="stSidebar"] div[style*="cursor: col-resize"] > div {
  background: #242424 !important;
  background-image: none !important;
  opacity: .45;
}

[data-testid="stFileUploaderDropzone"] {
  border: 1px dashed #454545;
  border-radius: var(--ui-radius);
  background: #0d0d0d;
  transition: border-color 110ms ease, background-color 110ms ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
  border-color: #777777;
  background: #121212;
}

[data-testid="stAlert"] {
  border: 1px solid #343434;
  border-radius: var(--ui-radius);
  background: #151515 !important;
  box-shadow: none !important;
  filter: grayscale(1) saturate(0);
}

[data-testid="stBadge"] { filter: grayscale(1) saturate(0); }

[data-testid="stSkeleton"] {
  border-radius: 6px;
  background: #1a1a1a;
  animation: ui-pulse 1.35s ease-in-out infinite alternate;
}

@keyframes ui-pulse { to { opacity: .52; } }

* { scrollbar-width: thin; scrollbar-color: #3b3b3b transparent; }
*::-webkit-scrollbar { width: 7px; height: 7px; }
*::-webkit-scrollbar-thumb { background: #3b3b3b; border-radius: 99px; }
*::-webkit-scrollbar-track { background: transparent; }

/* Navegación inferior compacta en móvil. */
@media (max-width: 768px) {
  .stMainBlockContainer,
  [data-testid="stMainBlockContainer"] {
    padding: 4rem .8rem calc(5.7rem + env(safe-area-inset-bottom)) !important;
  }

  .st-key-page_header {
    padding-top: .1rem;
    padding-bottom: .8rem;
    margin-bottom: .2rem;
  }

  .st-key-page_header h1 { font-size: 1.62rem; }

  [data-testid="stHorizontalBlock"]:has(> div [data-testid="stMetric"]) > div {
    flex: 1 1 100% !important;
    min-width: 100% !important;
  }

  div[data-testid="stMetric"] {
    min-height: 5.55rem;
    padding: .68rem .76rem;
  }

  section[data-testid="stSidebar"] {
    position: fixed !important;
    inset: auto 0 0 0 !important;
    z-index: 990;
    width: 100vw !important;
    min-width: 100vw !important;
    height: 4.85rem !important;
    overflow: hidden;
    border: 0;
    border-top: 1px solid #2a2a2a;
    border-radius: 0;
    background: #050505 !important;
    box-shadow: none !important;
  }

  section[data-testid="stSidebar"] > div:first-child {
    width: 100% !important;
    padding: .3rem .4rem .22rem !important;
    overflow: hidden !important;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
  section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    display: none !important;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
    position: absolute !important;
    inset: .28rem .35rem .12rem !important;
    width: auto !important;
    height: auto !important;
    margin: 0 !important;
    padding: 0 !important;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {
    display: flex;
    flex-direction: row;
    gap: .12rem;
    padding: 0 0 .15rem;
    overflow-x: auto;
    overflow-y: hidden;
    scroll-snap-type: x proximity;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] li {
    flex: 0 0 auto;
    scroll-snap-align: center;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: .04rem;
    width: 6.25rem;
    min-height: 4rem;
    margin: 0;
    padding: .28rem .22rem;
    border: 0;
    border-top: 1px solid transparent;
    border-radius: 3px;
    text-align: center;
    font-size: .6rem;
    line-height: 1.05;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    border-top-color: #ffffff;
    border-left-color: transparent;
    background: #111111;
  }

  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a p {
    display: -webkit-box !important;
    width: 100% !important;
    max-width: 100% !important;
    min-height: 1.25rem !important;
    margin: 0 !important;
    overflow: hidden !important;
    -webkit-box-orient: vertical !important;
    -webkit-line-clamp: 2 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    color: inherit !important;
    font-size: .56rem !important;
    line-height: 1.15 !important;
    text-align: center !important;
  }

  .st-key-sidebar_market_status,
  section[data-testid="stSidebar"] hr,
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
  }

  [data-baseweb="tab-list"], [role="tablist"] { flex-wrap: nowrap; }
  [data-baseweb="tab"], div[data-testid="stTab"] { min-width: max-content; }
}

@media (min-width: 1800px) {
  .stMainBlockContainer,
  [data-testid="stMainBlockContainer"] { max-width: 1700px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
  }
}
</style>
"""


def apply_premium_ui() -> None:
    """Aplica el sistema visual sin introducir estado ni efectos de negocio."""

    st.html(PREMIUM_CSS)
