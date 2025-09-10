import os
import shutil
import traceback
import PyPDF2
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")  
MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")
BASE_URL = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")

client = None
if API_KEY:
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    except Exception as e:
        client = None
        print("Warning: could not init OpenAI client:", e)

# Storage
ROOT = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(ROOT, "pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


CSS = r"""
/* General Reset */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: "Segoe UI", sans-serif;
}

body {
  background: #1e1e1e;
  color: #e6ccb2;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  display: flex;
  flex-direction: column;
} /* Landing Page */
.landing {
  height: 100vh;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 2rem;
}
.landing h1 {
  font-size: 3rem;
  margin-bottom: 1.5rem;
  color: #f5ebe0;
}
.landing p {
  font-size: 1.2rem;
  margin-bottom: 1rem;
  color: #e6ccb2;
  line-height: 1.6;
}
.enter-btn {
  margin-top: 2rem;
  padding: 0.8rem 2rem;
  background: #e6ccb2;
  color: #1e1e1e;
  border: none;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: bold;
  cursor: pointer;
  transition: 0.3s;
}
.enter-btn:hover {
  background: #d4b59e;
} /* Login Page / / Login Page Styling */
.login {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #212426; /* Dark background */
  color: #f9d3b4; /* Light text color */
  padding: 1rem;
}
.login h1 {
  font-size: 2.5rem;
  margin-bottom: 2rem;
  text-align: center;
}
.login input {
  width: 90%;
  max-width: 400px;
  padding: 1rem;
  margin: 0.5rem 0;
  font-size: 1.1rem;
  border: none;
  border-radius: 5px;
  background-color: #f9d3b4;
  color: #212426;
  outline: none;
}
.login button {
  width: 90%;
  max-width: 400px;
  padding: 1rem;
  margin-top: 1rem;
  font-size: 1.1rem;
  border: none;
  border-radius: 5px;
  background-color: #f9d3b4;
  color: #212426;
  cursor: pointer;
  transition: background-color 0.3s;
}
.login button:hover {
  background-color: #e0b89a;
}
.login .back-btn {
  background-color: #ff6b6b;
  color: #fff;
}
.login .back-btn:hover {
  background-color: #e63946;
} /* Chat Layout */
.chat-container {
  height: 100vh;
  width: 100%;
  display: flex;
  background: #2c2c2c;
} /* Sidebar */
.sidebar {
  width: 220px;
  background: #141414;
  color: white;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 1rem 0;
}
.sidebar .logo {
  font-size: 2rem;
  margin-bottom: 2rem;
}
.sidebar-btn {
  background: #0077b6;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 0.7rem 1rem;
  margin-bottom: 1rem;
  cursor: pointer;
  font-size: 1rem;
  width: 80%;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.sidebar-btn:hover {
  background: #005f8d;
}
.search-box {
  display: flex;
  align-items: center;
  background: #222;
  padding: 0.5rem;
  border-radius: 6px;
  width: 80%;
}
.search-box i {
  margin-right: 0.5rem;
  color: #aaa;
}
.search-box input {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  color: white;
} /* Modal (Settings) */
.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  justify-content: center;
  align-items: center;
}
.modal.hidden {
  display: none;
}
.modal-content {
  background: #2c2c2c;
  color: white;
  padding: 2rem;
  border-radius: 10px;
  width: 300px;
  text-align: center;
} /* Chat Main Area */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
} /* Header */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1a1a1a;
  color: #fff;
  padding: 1rem;
  font-size: 1.2rem;
  position: relative;
}
.chat-header .back-btn {
  position: absolute;
  left: 1rem;
  background: #ff6b6b;
  color: #fff;
  border: none;
  border-radius: 6px;
  padding: 0.5rem 1rem;
  cursor: pointer;
}
.chat-header .back-btn:hover {
  background: #e63946;
} /* Chat messages */
.chat-box {
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
}
.chat-message {
  margin: 0.5rem 0;
  padding: 0.8rem 1rem;
  border-radius: 18px;
  max-width: 75%;
}
.chat-message.user {
  background: #0077b6;
  color: #fff;
  margin-left: auto;
  border-bottom-right-radius: 4px;
}
.chat-message.bot {
  background: #e0e0e0;
  color: #333;
  margin-right: auto;
  border-bottom-left-radius: 4px;
}
.chat-message.bot h2 {
  font-size: 18px;
  margin-top: 10px;
  text-decoration: underline;
}
.chat-message.bot ul {
  margin-left: 20px;
  list-style-type: disc;
} /* Input area */
.chat-input {
  display: flex;
  padding: 0.5rem;
  background: #1a1a1a;
}
.chat-input input {
  flex: 1;
  padding: 0.6rem;
  border: none;
  border-radius: 6px;
  margin-right: 0.5rem;
}
.chat-input button {
  padding: 0.6rem 1rem;
  border: none;
  border-radius: 6px;
  background: #4cafef;
  color: white;
  cursor: pointer;
}
.chat-input button:hover {
  background: #3399ff;
}
.chat-input textarea {
  flex: 1;
  padding: 0.6rem;
  border: none;
  border-radius: 6px;
  margin-right: 0.5rem;
  resize: none; /* user cannot drag resize */
  max-height: 200px; /* prevent it from growing too tall */
  line-height: 1.4;
  background: #fff;
  color: #000;
} /* Quick Replies */
.quick-replies {
  display: flex;
  justify-content: center;
  gap: 1rem;
  padding: 0.8rem;
  background: #1a1a1a;
}
.quick-replies button {
  background: #f4a261;
  color: #fff;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  cursor: pointer;
  transition: 0.3s;
}
.quick-replies button:hover {
  background: #e76f51;
}
.upload-btn {
  cursor: pointer;
  display: flex;
  justify-content: center;
  align-items: center;
}
.upload-btn input {
  display: none;
} /* Hidden */
.hidden {
  display: none;
}
"""

# Helpers for PDFs
def load_pdfs_text(folder_path=UPLOAD_FOLDER):
    all_text = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        txt = page.extract_text()
                        if txt:
                            all_text.append(txt.strip())
            except Exception as e:
                print("Error reading PDF:", pdf_path, e)
    return "\n".join(all_text) if all_text else "No text found"

SYSTEM_PROMPT = """
You are Exam Assistant, an AI study helper.
Your role is to give clear, structured, and exam-ready answers using uploaded PDFs as reference.
Rules:
1. Use Markdown formatting for structure and clarity.
   - Use headings (## Definition, ## Explanation, etc.)
   - Use bullet points (- or 1.) for lists
   - Use bold only for key terms, not decoration
   - Do not use italics or unnecessary symbols
2. Keep answers exam-focused, neat, and easy to copy into answer sheets.
3. Do not greet the user with casual phrases. Do not make the answer sound like a quiz.
4. Keep the tone professional, concise, and focused, but easy to understand.
Important:
- Analyze the uploaded PDFs carefully and base answers on them.
- Build explanations step by step with headings and bullet points.
- Always end with a small follow-up question to keep the student engaged.
"""

def call_model_with_context(user_message):
    pdf_context = load_pdfs_text()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": f"Context from uploaded PDFs:\n{pdf_context}"},
        {"role": "user", "content": user_message},
    ]
    # If client exists, call model; else fallback
    if client:
        try:
            response = client.chat.completions.create(model=MODEL, messages=messages)
            try:
                return response.choices[0].message.content
            except Exception:
                try:
                    return response["choices"][0]["message"]["content"]
                except Exception:
                    return str(response)
        except Exception as e:
            print("Model call error:", e, traceback.format_exc())
            return f"⚠ Error calling model: {e}"
    else:
        return f"(No API key configured) Echo: {user_message}"

def send_message(user_input, chat_history):
    chat_history = chat_history or []
    user_input = (user_input or "").strip()
    if not user_input:
        return chat_history, ""
    try:
        # call model
        reply = call_model_with_context(user_input)
    except Exception as e:
        reply = f"⚠ Internal error: {e}"
    chat_history.append((user_input, reply))
    return chat_history, ""

def upload_pdf(file):
    if not file:
        return "No file provided"
    try:
        src = getattr(file, "name", None) or file
        filename = os.path.basename(src)
        dst = os.path.join(UPLOAD_FOLDER, filename)
        shutil.copy(src, dst)
        return f"✅ File {filename} uploaded successfully!"
    except Exception as e:
        return f"⚠ Upload failed: {e}"

def new_chat():
    return []

def show_history(chat_history_global):
    if not chat_history_global:
        return "No past conversations yet."
    lines = []
    for i, (u, b) in enumerate(chat_history_global, 1):
        lines.append(f"**Q{i}.** {u}\n\n**A{i}.**\n{b}\n\n---\n")
    return "\n".join(lines)

with gr.Blocks(css=CSS, title="Exam Assistant Chatbot") as demo:
    gr.HTML("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">""")

    # top-level layout: sidebar + main area
    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.HTML("<div class='sidebar'><div class='logo'><i class='fa-solid fa-graduation-cap'></i></div></div>")
            new_btn = gr.Button("New Chat", elem_classes="sidebar-btn")
            upload_btn = gr.File(file_count="single", label="Upload PDF", file_types=[".pdf"], elem_id="file-upload", visible=True)
            history_btn = gr.Button("History", elem_classes="sidebar-btn")
            # invisible element to show upload output
            upload_out = gr.Textbox(value="", visible=False)
        with gr.Column(scale=3):
            gr.HTML("<div class='chat-header'><h2>Exam Assistant Chatbot</h2></div>")
            chatbot = gr.Chatbot(elem_classes="chat-box", label=None)
            with gr.Row(elem_classes="chat-input"):
                inp = gr.Textbox(placeholder="Type your question… (Shift+Enter = new line)", elem_id="user-input")
                send = gr.Button(elem_classes="sidebar-btn", value="Send")  # styling reuse
            with gr.Row(elem_classes="quick-replies"):
                quick1 = gr.Button("Generate Questions")
                quick2 = gr.Button("Summarize")
            # Area to display history markdown when pressed
            history_md = gr.Markdown("")

    # State to keep global history
    history_state = gr.State([])  # list of (user, bot) tuples

    # Wiring: actions
    send.click(fn=send_message, inputs=[inp, history_state], outputs=[chatbot, inp]).then(lambda *args: None, None, None)
    new_btn.click(fn=new_chat, outputs=[chatbot, history_state])
    upload_btn.upload(fn=upload_pdf, inputs=[upload_btn], outputs=[upload_out])
    history_btn.click(fn=lambda hist: show_history(hist), inputs=[history_state], outputs=[history_md])

    quick1.click(fn=lambda hist: send_message("Generate exam questions", hist), inputs=[history_state], outputs=[chatbot, inp])
    quick2.click(fn=lambda hist: send_message("Summarize this topic", hist), inputs=[history_state], outputs=[chatbot, inp])

# Launch Gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
