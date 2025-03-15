import os
import fitz  # PyMuPDF untuk membaca PDF
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq  # Groq API untuk NLP

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Fungsi untuk membaca PDF
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Load dan split dokumen
pdf_text = load_pdf("kebijakan_capil.pdf")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
docs = text_splitter.split_text(pdf_text)

# Inisialisasi embeddings
embeddings = HuggingFaceEmbeddings()

# Simpan embeddings di FAISS
vectorstore = FAISS.from_texts(docs, embeddings)
retriever = vectorstore.as_retriever()

# Inisialisasi model Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Template Prompt
TEMPLATE = """
Anda adalah asisten informasi untuk DISDUKCAPIL Kota Sorong. Jawablah pertanyaan pengguna berdasarkan konteks berikut.
Jika Anda tidak tahu jawabannya, cukup katakan saya tidak mengerti maksud anda.
Jawaban harus singkat, tidak lebih dari 2 kalimat.

Konteks: {context}
Pertanyaan: {question}
Jawaban:
"""

def query_groq(prompt):
    """ Mengirim prompt ke Groq API dan mendapatkan respons. """
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Aku adalah asisten informasi untuk DISDUKCAPIL Kota Sorong."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_rag_response(user_input):
    """ Menggunakan FAISS untuk mengambil informasi dan menggabungkan dengan LLM dari Groq. """
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    final_prompt = TEMPLATE.format(context=context, question=user_input)
    return query_groq(final_prompt)

@app.route("/")
def index():
    """ Menampilkan halaman utama. """
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """ API endpoint untuk menangani chat. """
    user_input = request.json.get("message", "").strip().lower()
    
    # Daftar sapaan yang dikenali
    greetings = ["halo", "hai", "hello", "hi", "selamat pagi", "selamat siang", "selamat sore", "selamat malam"]

    # Jika pengguna menyapa, sistem akan membalas dengan perkenalan diri
    if user_input in greetings:
        return jsonify({"response": "👋 Halo! Saya adalah chatbot DISDUKCAPIL Kota Sorong. Saya siap membantu Anda dengan informasi kependudukan dan pencatatan sipil."})
    
    # Jika bukan sapaan, gunakan sistem RAG untuk mencari jawaban
    response = get_rag_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
