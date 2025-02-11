from langchain_text_splitters import TokenTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from commons import init_model, init_embedding

model = init_model()
embedding = init_embedding()
text_splitter = TokenTextSplitter.from_tiktoken_encoder(model_name="gpt-4o-mini")

def get_vectorstore(text_chunks):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vectorstore

def generate_itinerary(location, interests):
    prompt = f"""You are a travel agent creating a detailed itinerary for the tourists spots/destinations in the Philippines.

    Location: {location}
    Interests: {interests}

    Create a multi-day itinerary. For each day, suggest specific activities, restaurants (if known), and points of interest. Be creative and provide a variety of options. Consider the user's interests when making suggestions. Provide estimated timeframes for activities. Format the itinerary clearly and concisely.

    Example:

    **Day 1: Arrival and City Exploration**

    * Morning (9:00 AM): Arrive at {location}, transfer to hotel.
    * Afternoon (1:00 PM): Lunch at [Restaurant Name] (Cuisine type).
    * ...

    **Day 2: ...**
    ...
    """

    itinerary = model(prompt)
    return itinerary.content  # Extract the text content

def main():
    st.set_page_config(page_title="AI Travel Assistant - Philippines", page_icon=":airplane:")
    st.header("AI Travel Assistant - Philippines :flag-ph:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history FIRST
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.subheader("Your Preferences")
        location = st.text_input("Enter your desired location", placeholder= "Enter location here")
        interest = st.multiselect("Interests (optional)", ["Dive", "Nature", "Sun and Beach", "Adventure", "Food and Hospitality", "Events and Culture"])

        if st.button("Generate Itinerary"):
            if location:
                with st.spinner("Generating Itinerary..."):
                    itinerary = generate_itinerary(location, interest)

                    # Update chat history and display immediately
                    st.session_state.messages.append({"role": "assistant", "content": itinerary})
                    st.rerun()  # Force Streamlit to rerun to update the chat display

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
        else:
            st.warning("Please process the documents first.")

if __name__ == "__main__":
    main()