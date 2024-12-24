import streamlit as st
from config import settings
from utils.retrievers.FusionRetriever import FusionRetriever
from utils.prompts.Prompts import Prompt
from utils.llms.GigaModel import GigaApi
from utils.dto.FusionDTO import FusionDTO
from typing import List

@st.cache_resource
def load_model():
    llm = GigaApi()
    fusion_retriever = FusionRetriever(settings=settings)
    return llm, fusion_retriever

def make_prompt(dtos: List[FusionDTO]) -> str:
    gen_prompt = Prompt.generation_prompt
    few_shot_prompts = [Prompt.few_shot.format(
        law=elem.law_number, title=elem.title, page_first=elem.page_first, page_last=elem.page_last, context=elem.text
        ) for elem in dtos]
    return gen_prompt + "\n".join(few_shot_prompts) + "\n" + "Ответ:", few_shot_prompts[:3]

def make_retrieve(llm, retriever, query) -> List[FusionDTO]:
    # llm, retriever = load_model()
    dtos = retriever.retrieve(query)
    full_prompt, contexts = make_prompt(dtos)
    answer = llm.inference(full_prompt)
    return answer, contexts


if 'history' not in st.session_state:
    st.session_state['history'] = []

        
st.title("Консультант государственных закупок")
        
llm, retriever = load_model()

user_question = st.text_input("Введите свой запрос:", "")


if st.button('Отправить'):
    if user_question:
        answer, contexts = make_retrieve(llm, retriever, user_question)
        
        st.subheader("Вопрос")
        st.write(user_question)
        st.subheader("Ответ GigaChat")
        st.write(answer)
        st.subheader("Использованные отрывки документов")
        
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white;">{contexts[0]}</div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white;">{contexts[1]}</div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="background-color: #333; padding: 10px; border-radius: 5px; color: white;">{contexts[2]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        context = f"{contexts[0]}\n\n{contexts[1]}\n\n{contexts[2]}"

        st.session_state['history'].append((user_question, answer, context))
    else:
        st.error("Пожалуйста, введите вопрос")

        
st.sidebar.title("История запросов")
for idx, (question, answer, context) in enumerate(reversed(st.session_state['history']), start=1):
    with st.sidebar.expander(f"Вопрос {idx}: {question}"):
        st.write(f"**Ответ:** {answer}")
        st.write(f"**Контекст:** {context}") 