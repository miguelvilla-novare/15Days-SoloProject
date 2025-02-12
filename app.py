from langchain_text_splitters import TokenTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from commons import init_model, init_embedding
import pandas as pd

model = init_model()
embedding = init_embedding()
calabarzon_provinces = ["Cavite", "Laguna", "Batangas", "Rizal", "Quezon"]

df = pd.read_csv("/home/franciscovilla/GenAI/15Days-SoloProject/src/calabarzon-touristspots.csv")

# Create and store vectorstore in session state (only once)
if "vectorstore" not in st.session_state:
    texts = df["Destination-Activities"].tolist()
    metadatas = df.drop("Destination-Activities", axis=1).to_dict(orient="records")
    vectorstore = FAISS.from_texts(texts, embedding, metadatas=metadatas)
    st.session_state.vectorstore = vectorstore
    
def main():
    st.set_page_config(page_title="AI Travel Assistant", page_icon=":airplane:")
    st.header("AI Travel Assistant- CALABARZON")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.subheader("Your Preferences")
        location = st.selectbox("Enter your desired location", calabarzon_provinces)
        interest = st.multiselect("Interests (optional)", ["Nature", "Food", "Adventure"])

        if st.button("Generate Itinerary"):
            if location:
                with st.spinner("Generating Itinerary..."):
                    query_embedding = embedding.embed_query(f"{location} {' '.join(interest)}")  # Use the initialized embedding
                    results = st.session_state.vectorstore.similarity_search_with_score_by_vector(query_embedding, k=5)

                    prompt = f"""You are a travel agent creating a detailed itinerary. Use the following template for the itinerary and strictly follow the user's location and interests
                                Fill every section in the example below with specific details for each day and activity.  
                                Be creative and consider the user's interests when making suggestions.  If a particular section (like Lunch or Evening) is not applicable, you can omit it.
                                Provide realistic timeframes.
            
                        Example:
                        Location: {location}
                        Interests: {interest}
                        
                        Day 1: Arrival and Exploring Liliw
                        
                        Morning (9:00 AM - 12:00 PM)
                        Activity: Liliw Footwear Shopping
                        Location: Liliw, Laguna
                        Description: Start your day in the charming town of Liliw... (rest of the description)

                        Lunch (12:30 PM - 1:30 PM)
                        Restaurant: Arabela
                        Description: Savor delicious pasta dishes... 
                        
                        (rest of the itinerary)
            
                        """
                    for doc, score in results:
                        metadata = doc.metadata
                        prompt += f"- **{doc.page_content}** (Province: {metadata['Province']}, Description: {metadata['Description']})\n"

                    itinerary = model(prompt).content  # Use your LLM from commons.py

                    st.session_state.messages.append({"role": "assistant", "content": itinerary})
                    st.rerun()

            else:
                st.warning("Please enter a location.")

    if user_input := st.chat_input("Ask questions about your itinerary"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if "vectorstore" in st.session_state and st.session_state.vectorstore is not None:
            vectorstore = st.session_state.vectorstore
            qa_chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=vectorstore.as_retriever())
            with st.spinner("Generating Answer"):
                answer = qa_chain.run(user_input)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)

if __name__ == "__main__":
    main()
    
