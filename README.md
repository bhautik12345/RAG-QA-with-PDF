# 📄💬 Conversational RAG with PDF Uploads & Chat History

**Conversational RAG** is an intelligent PDF assistant that allows you to upload PDF files and have dynamic conversations with their content. It uses Retrieval-Augmented Generation (RAG) techniques to deliver accurate and context-aware answers from your documents — all while preserving your chat history for a seamless experience.

---

## 🚀 Features

- 📁 Upload multiple PDF files
- 💬 Ask questions and get answers from your PDFs
- 🧠 Keeps chat history for context
- ⚠️ Detects unextractable PDFs (e.g., scanned or protected)

---

## 📎 Usage Tips

- 🔹 Upload up to 10 MB per file
- 🔹 Use clear, specific questions for best results
- 🔹 Some PDFs may not support text extraction

---

## ▶️ Run Locally

```bash
git clone https://github.com/bhautik12345/RAG-QA-with-PDF.git
cd RAG-QA-with-PDF
pip install -r requirements.txt
streamlit run app.py
