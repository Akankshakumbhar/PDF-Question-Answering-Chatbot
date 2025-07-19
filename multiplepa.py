import streamlit as st
import google.generativeai as genai

# 🔐 Configure Gemini API
API_KEY = "AIzaSyAhPN7G-Mrv2cPCKP1VYuXL5ffFopwtiAM"  # Replace with your real Gemini 1.5 Pro API key
genai.configure(api_key=API_KEY)

# 📘 Use Gemini 1.5 Pro for native PDF analysis
model = genai.GenerativeModel("gemini-1.5-flash")
#model = genai.GenerativeModel("gemini-2.5-pro-latest")

# 🚀 Streamlit App
def main():
    st.set_page_config(page_title=" Ask PDF (Gemini-native)", layout="centered")
    st.title(" Ask Anything About the Documents")

    # ✅ Predefined PDF files
    predefined_pdfs = [
        "startupai-financial-report-v2.pdf",
        "microservice arcticture.pdf",
        "basic-understanding-of-a-companys-financials.pdf",
        "qpm help.pdf",
        "orcerstractionframework.pdf"
    ]

    # 🧠 Load predefined PDFs as binary content
    pdf_contents = []
    for file_path in predefined_pdfs:
        try:
            with open(file_path, "rb") as f:
                pdf_contents.append({
                    "mime_type": "application/pdf",
                    "data": f.read()
                })
        except FileNotFoundError:
            st.error(f"❌ Could not find the file: {file_path}")
            return

    # 💬 Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✍️ Input field for user question
    question = st.text_input("Enter your question about the document(s)")

    # 🧠 Generate an answer from predefined PDFs
    def generate_answer_from_pdfs(pdf_contents, question):
        try:
            prompt = f"""
            You are document analysis expert.
            Analyze the uploaded documents and answer this question:
            {question}

            Also, please include the **page number(s)** and the **document name** if possible.
            """
            # Add prompt to the list of contents
            pdf_contents_with_prompt = pdf_contents + [prompt]

            # Ask Gemini
            response = model.generate_content(pdf_contents_with_prompt)
            return response.text.strip()
        except Exception as e:
            return f"❌ Error from Gemini: {e}"

    # If there's a question, generate the response
    if question:
        with st.spinner("Analyzing documents and generating response..."):
            answer = generate_answer_from_pdfs(pdf_contents, question)
            st.session_state.chat_history.append({"question": question, "answer": answer})
            st.success("✅ Answer generated!")

    # 💬 Display chat history
    if st.session_state.chat_history:
        st.subheader("🧾 Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.markdown("---")

if __name__ == "__main__":
    main()
