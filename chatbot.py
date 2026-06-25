#!/usr/bin/env python3
"""
GeoCode-Mistral Chatbot Interface
A streaming Gradio UI for the fine-tuned Mistral-7B geospatial code model.

- gr.Blocks + gr.Chatbot with full conversation history
- Token-by-token streaming (TextIteratorStreamer + Thread)
- Domain-aware system prompts (GEE / ArcPy / QGIS / GeoPandas / auto-detect)
- Mistral [INST] prompt format (matches showcase_inference.py)
- Professional dark UI (GitHub palette)
"""

import json
import threading
from datetime import datetime

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_PATH = "./models/final_model"
LOG_FILE = "outputs/chatbot_history.jsonl"

GITHUB_URL = "https://github.com/SedatAliMengi/GeoCode-Mistral"
PAPER_URL = "https://doi.org/10.1016/j.jag.2025.104456"

# Generation defaults (proven values from showcase_inference.py)
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_NEW_TOKENS = 400
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.2

# --------------------------------------------------------------------------- #
# Domain definitions: each injects a different system prompt
# --------------------------------------------------------------------------- #
AUTO_DETECT = "Auto-detect"

DOMAINS = {
    AUTO_DETECT: (
        "You are an expert in geospatial analysis and programming across "
        "multiple platforms and languages. Analyze the user's request, choose "
        "the most appropriate library or platform, and produce correct, "
        "runnable code with a brief explanation."
    ),
    "Google Earth Engine": (
        "You are an expert in Google Earth Engine. Write correct, runnable "
        "Earth Engine code (JavaScript Code Editor API unless the user asks "
        "for the Python `ee` API). Use proper image collections, filtering, "
        "reducers, and Map/Export calls."
    ),
    "ArcPy": (
        "You are an expert in Esri ArcPy for ArcGIS Pro. Write correct, "
        "runnable Python using the `arcpy` module, with valid tool names, "
        "parameters, environment settings, and cursors where appropriate."
    ),
    "QGIS (PyQGIS)": (
        "You are an expert in PyQGIS, the Python API for QGIS. Write correct, "
        "runnable PyQGIS code using QgsProject, QgsVectorLayer, QgsRasterLayer, "
        "and the processing framework where appropriate."
    ),
    "GeoPandas & Python GIS": (
        "You are an expert in the Python geospatial stack (GeoPandas, Shapely, "
        "rasterio, pyproj, Fiona). Write correct, runnable Python with proper "
        "CRS handling, geometry operations, and I/O."
    ),
}

# Keyword hints for Auto-detect mode
DOMAIN_KEYWORDS = {
    "Google Earth Engine": [
        "earth engine", "gee", "ee.", "imagecollection", "sentinel", "landsat",
        "map.addlayer", "ee.image", "qualitymosaic", "export.image",
    ],
    "ArcPy": ["arcpy", "arcgis", "arctoolbox", "esri", ".sde", "feature class"],
    "QGIS (PyQGIS)": [
        "qgis", "pyqgis", "qgsproject", "qgsvectorlayer", "qgsrasterlayer",
    ],
    "GeoPandas & Python GIS": [
        "geopandas", "geodataframe", "shapely", "rasterio", "fiona", "pyproj",
        "shapefile", "geojson", "buffer", "crs",
    ],
}

# One example prompt per domain (label -> (domain, prompt))
EXAMPLES = [
    (
        "🛰️ GEE — NDVI",
        "Google Earth Engine",
        "Write Google Earth Engine code to compute NDVI from a Sentinel-2 "
        "image collection over a region, take the greenest pixel composite, "
        "and add it to the map.",
    ),
    (
        "🗺️ ArcPy — Buffer",
        "ArcPy",
        "Write ArcPy code to create a 500 meter buffer around a feature class "
        "and save the result to a file geodatabase.",
    ),
    (
        "🧭 PyQGIS — Load layer",
        "QGIS (PyQGIS)",
        "Write PyQGIS code to load a shapefile as a vector layer, add it to "
        "the current project, and apply a simple categorized symbology.",
    ),
    (
        "🐍 GeoPandas — Reproject",
        "GeoPandas & Python GIS",
        "Write Python code with GeoPandas to read a shapefile, reproject it to "
        "EPSG:3857, and compute the area of each polygon in square kilometers.",
    ),
    (
        "📡 Rasterio — Read GeoTIFF",
        "GeoPandas & Python GIS",
        "Write Python code using rasterio to open a GeoTIFF, read the first "
        "band, and print the CRS and pixel resolution.",
    ),
]

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
print("Loading model and tokenizer from", MODEL_PATH, "...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully.")


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def detect_domain(message: str) -> str:
    """Pick the most likely domain from the message text (Auto-detect mode)."""
    text = message.lower()
    best, best_score = AUTO_DETECT, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best, best_score = domain, score
    return best


def build_prompt(message: str, domain: str, history: list) -> str:
    """
    Build a Mistral [INST] prompt from the system prompt, prior turns, and the
    new message. `history` is a list of {"role", "content"} dicts for completed
    turns (excluding the current message). The training "END" marker is handled
    internally here and never surfaced to the user.

    Format:
        <s>[INST] {system}\n\n{user1} [/INST] {assistant1}</s>[INST] {user2} [/INST] ...
    """
    if domain == AUTO_DETECT:
        domain = detect_domain(message)
    system_prompt = DOMAINS.get(domain, DOMAINS[AUTO_DETECT])

    # Pair up prior turns (user, assistant)
    turns = []
    pending_user = None
    for msg in history:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            turns.append((pending_user, msg["content"]))
            pending_user = None

    parts = []
    first = True
    for user_text, assistant_text in turns:
        if first:
            parts.append(f"<s>[INST] {system_prompt}\n\n{user_text} [/INST] {assistant_text}</s>")
            first = False
        else:
            parts.append(f"[INST] {user_text} [/INST] {assistant_text}</s>")

    if first:
        # No prior turns: system prompt goes with the current message
        parts.append(f"<s>[INST] {system_prompt}\n\n{message} [/INST]")
    else:
        parts.append(f"[INST] {message} [/INST]")

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Streaming generation
# --------------------------------------------------------------------------- #
def log_interaction(message: str, response: str, domain: str) -> None:
    """Silently append the interaction to a JSONL log (best effort)."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "domain": domain,
                "message": message,
                "response": response,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def respond(message, history, domain, temperature, max_new_tokens):
    """Gradio handler: stream the model response into the chat history."""
    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    prompt = build_prompt(message, domain, history[:-2])
    # With device_map="auto" + CPU offload, model.device can be "cpu" while the
    # compute device is cuda:0. Put inputs on the GPU when one is available.
    input_device = "cuda" if torch.cuda.is_available() else model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial = ""
    for token in streamer:
        partial += token
        history[-1]["content"] = partial
        yield history, ""

    log_interaction(message, partial.strip(), domain)
    history[-1]["content"] = partial.strip()
    yield history, ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
:root { color-scheme: dark; }
.gradio-container {
    background: #0D1117 !important;
    color: #C9D1D9 !important;
    max-width: 1100px !important;
}
#header {
    text-align: center;
    padding: 18px 0 6px 0;
}
#header h1 {
    color: #F0F6FC;
    font-size: 1.9rem;
    margin: 0;
    letter-spacing: -0.5px;
}
#header .subtitle {
    color: #8B949E;
    font-size: 0.95rem;
    margin-top: 4px;
}
.pills { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin: 12px 0; }
.pill {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #C9D1D9;
}
.pill b { color: #3FB950; }
.links { text-align: center; margin: 6px 0 14px 0; font-size: 0.85rem; }
.links a { color: #3FB950; text-decoration: none; margin: 0 10px; }
.links a:hover { text-decoration: underline; }
#chatbot { background: #0D1117 !important; border: 1px solid #30363D !important; }
.gr-button-primary, button.primary {
    background: #238636 !important;
    border-color: #2EA043 !important;
    color: #FFFFFF !important;
}
.gr-button-primary:hover, button.primary:hover { background: #2EA043 !important; }
footer { display: none !important; }
"""

PILLS_HTML = """
<div class="pills">
  <span class="pill"><b>60%</b> task success</span>
  <span class="pill"><b>Mistral-7B</b> base</span>
  <span class="pill"><b>QLoRA + LoRA</b> two-stage</span>
  <span class="pill"><b>8GB VRAM</b> RTX 4070 Laptop</span>
</div>
"""

HEADER_HTML = f"""
<div id="header">
  <h1>🌍 GeoCode-Mistral</h1>
  <div class="subtitle">Fine-tuned Mistral-7B for geospatial code generation —
    Google Earth Engine · ArcPy · QGIS · GeoPandas</div>
</div>
{PILLS_HTML}
<div class="links">
  <a href="{GITHUB_URL}" target="_blank">⭐ GitHub Repository</a>
  <a href="{PAPER_URL}" target="_blank">📄 Original Paper (Hou et al., 2025)</a>
</div>
"""

with gr.Blocks(title="GeoCode-Mistral") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Row():
        domain_selector = gr.Dropdown(
            choices=list(DOMAINS.keys()),
            value=AUTO_DETECT,
            label="Domain",
            scale=2,
        )

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=460,
        avatar_images=(None, None),
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask for geospatial code — e.g. 'Compute NDVI from Sentinel-2 in Earth Engine'",
            show_label=False,
            scale=8,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear conversation", scale=1)

    gr.Markdown("### Examples")
    with gr.Row():
        for label, ex_domain, ex_prompt in EXAMPLES:
            btn = gr.Button(label, size="sm")
            btn.click(
                fn=lambda d=ex_domain, p=ex_prompt: (d, p),
                outputs=[domain_selector, msg],
            )

    with gr.Accordion("⚙️ Generation settings", open=False):
        temperature = gr.Slider(0.1, 1.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature")
        max_new_tokens = gr.Slider(64, 1024, value=DEFAULT_MAX_NEW_TOKENS, step=32, label="Max new tokens")

    # Wiring
    submit_args = dict(
        fn=respond,
        inputs=[msg, chatbot, domain_selector, temperature, max_new_tokens],
        outputs=[chatbot, msg],
    )
    msg.submit(**submit_args)
    send_btn.click(**submit_args)
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("GeoCode-Mistral chatbot starting...")
    print("Open in browser: http://localhost:7860")
    print("Press Ctrl+C to quit.")
    print("=" * 50 + "\n")
    demo.launch(share=False, theme=gr.themes.Base(), css=CUSTOM_CSS)
