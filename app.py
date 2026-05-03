import os
import socket
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from utils import get_vector_store
from ingest import add_document
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'xlsx', 'xls'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
vector_store = None
retriever = None
rag_chain = None

def init_rag():
    global vector_store, retriever, rag_chain
    try:
        print("Loading RAG system...")
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the question based on the provided context.
If the answer is not found in the context, say: "I couldn't find information about this in the uploaded documents."
Be concise and answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG system loaded successfully!")
        return True
    except Exception as e:
        print(f"RAG init error: {e}")
        return False

# Initialize on startup
init_rag()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'rag_ready': rag_chain is not None})

@app.route('/chat', methods=['POST'])
def chat():
    # Robust JSON parsing
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Invalid request format!'}), 400

    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Sawaal khali hai!'}), 400

    if rag_chain is None:
        return jsonify({'error': 'RAG system abhi load nahi hua. Page refresh karein!'}), 503

    try:
        docs = retriever.invoke(question)
        answer = rag_chain.invoke(question)
        sources = list(set([
            os.path.basename(doc.metadata.get("source", ""))
            for doc in docs if doc.metadata.get("source")
        ]))
        return jsonify({'answer': answer, 'sources': sources})
    except Exception as e:
        error_msg = str(e)
        print(f"Chat error: {error_msg}")
        if "HF_TOKEN" in error_msg or "401" in error_msg or "403" in error_msg:
            return jsonify({'error': 'HuggingFace token invalid hai! Render pe HF_TOKEN check karein.'}), 500
        return jsonify({'error': f'Error: {error_msg}'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'Koi file nahi mili!'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Sirf PDF, Word (docx), Excel (xlsx) allowed hai!'}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        add_document(filepath)
        # Refresh retriever after new doc
        global vector_store, retriever
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return jsonify({'success': True, 'message': f'"{filename}" successfully RAG mein add ho gaya!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    is_production = os.environ.get('RENDER', False)

    if not is_production:
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"\n{'='*50}")
            print(f"  RAG Chat Server chal raha hai!")
            print(f"  PC pe    : http://localhost:{port}")
            print(f"  Mobile pe: http://{local_ip}:{port}")
            print(f"{'='*50}\n")
        except:
            pass

    app.run(host='0.0.0.0', port=port, debug=False)
