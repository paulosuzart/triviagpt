
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import RetryWithErrorOutputParser
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets.oai.OPENAI_API_KEY

class Question(BaseModel):
    question: str = Field(description='The question of the trivia')
    answer: str = Field(description='The correct answer')
    ops: list[str] = Field(description='Options')
    link: str = Field(description='A link to a content related to the answer')

st.title('TriviaGPT')

subject = st.text_input(label='Which topic you wanna have a TriviaGPT?', max_chars=80, help='Summer in poland')

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Question)

llm = OpenAI(temperature=0.9, model_name='text-davinci-003')
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm)

def fetch_question(_subject) -> Question:
    prompt = PromptTemplate(
        input_variables=["subject"],
        template="""
        Prepare a trivia game. {format_instructions}.
        Prepare a trivia about the Subject: {subject}.
        Bring only 1 questions. 
        Give three alternatives to each question where one is the correct. Keep answers as short as possible.
        No sexual or minor than 18 years subjects must be brought up. Questions in english only.
    """,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    q = prompt.format_prompt(subject=subject)
    q2 = prompt.format(subject=_subject)
    print(q)

    result = llm(q2)
    try:
        return retry_parser.parse_with_prompt(result, q)
    except:
        st.warning('Something went wrong. Please try again after refreshing.')
    
# Initialize the state
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'q' not in st.session_state:
    st.session_state.q = None
if 'option' not in st.session_state:
    st.session_state.option = '-'

# User has hit play
if st.button('Play'):
    st.session_state.playing = True
    q = fetch_question(subject)
    if len(q.ops) > 0:
        st.session_state.q = q

# Game is in place
if st.session_state.q:
    st.write('Here we go')
    st.write(st.session_state.q.question)
    options = ['-'] + st.session_state.q.ops
    option = st.selectbox('Pick one', options)

    if option == '-':
        st.stop()
    if option == st.session_state.q.answer:
        st.success('Bingo!')
        st.snow()
    else:
        st.error('Ooops, not this time')
