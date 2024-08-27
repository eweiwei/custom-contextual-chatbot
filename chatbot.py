from langchain.chains import ConversationalRetrievalChain, LLMChain

from langchain.prompts import PromptTemplate

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_aws.chat_models import ChatBedrock

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores import FAISS

import boto3
import os.path
import streamlit as st 

import json
import base64
from PIL import Image
from io import BytesIO

client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)

embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=client)

docs = []

processed_files = set()

context_prompt_template = '''
You are an AI assistant and expert with immense knowledge and experience in every field. 
Answer my questions based on your knowledge, the context wrapped in <context> XML tags, 
and our older conversation wrapped in <history> XML tags. You should first refer to the 
documents provided in the context to formulate your response. However, if the answer cannot 
be found in the context, use your own immense knowledge to answer the question to the best 
of your ability and state that the answer was not found in the context. Be sure to cite the 
specific document within the context that you used to formulate your answer by mentioning 
the name or title of the document you are pulling from. The question will be wrapped in 
<question> XML tags. Do not include XML tags in your answer.

\n<context>{context}</context>\n

Given the following conversation and a follow up question, answer the question.

\n<history>{chat_history}</history>\n

\n<question>{question}</question>\n

Helpful answer: 
'''

prompt_template = '''
You are an AI assistant and expert with immense knowledge and experience in every field. 
Answer my questions based on your knowledge and our older conversation wrapped in <history> 
XML tags. The question will be wrapped in <question> XML tags. Do not include XML tags in 
your answer.

\nGiven the following conversation and a follow up question, answer the question.

\n<history>{chat_history}</history>\n

\n<question>{question}</question>\n

Helpful answer: 
'''

CONTEXT_PROMPT = PromptTemplate(
    template=context_prompt_template, input_variables=["context", "chat_history", "question"]
)

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=['chat_history', 'question']
)

def process_files(uploaded_files):
    count = 0

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state['files']: 
            current_dir = os.getcwd()

            files_folder = os.path.join(current_dir, "rag")
            os.makedirs(files_folder, exist_ok=True)    
            file_path = os.path.join(files_folder, file_name)
   
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            loader = TextLoader(file_path=file_path)
            doc = loader.load()
            docs.extend(doc)

            st.session_state['files'].append(file_name)
            os.remove(file_path)

            count += 1
    
    if count > 0:
        save_vectors()

def save_vectors():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 100, separators = ["\n\n","\n"," ",""]) 
    text = text_splitter.split_documents(documents=docs) 

    vectorstore = FAISS.from_documents(text, embeddings)
    vectorstore.save_local('vectors')
    print('Embeddings saved')

def load_vectors():
    return FAISS.load_local('vectors', embeddings, allow_dangerous_deserialization=True)

def get_response(query, chat_history, llm, context):
    if context:
        loaded_vectors = load_vectors()

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=loaded_vectors.as_retriever(
                search_type='similarity', search_kwargs={'k': 3}
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': CONTEXT_PROMPT}
        )

        answer = qa.invoke({'question': query, 'chat_history': chat_history})

        return answer['answer']
    else:
        qa = LLMChain(
            llm=llm,
            prompt=PROMPT
        )

        return qa.invoke({'question': query, 'chat_history': chat_history})['text']


def load_llm(model_id, max_tokens, temperature):
    return ChatBedrock(
        model_id=model_id,
        client=client,
        model_kwargs={'max_tokens': max_tokens, 'temperature': temperature}
    )

def get_response_image_from_payload(response): 
    payload = json.loads(response.get('body').read()) 
    images = payload.get('artifacts')
    image_data = base64.b64decode(images[0].get('base64')) 

    return BytesIO(image_data) 

def get_image_response(prompt_content, model_id): 

    request_body = json.dumps({"text_prompts":
                               [ {"text": prompt_content } ], #prompts to use
                               "cfg_scale": 9, #how closely the model tries to match the prompt
                               "steps": 50, }) #number of diffusion steps to perform

    response = client.invoke_model(body=request_body, modelId=model_id) 

    img_io = get_response_image_from_payload(response)

    img_io.seek(0)
    
    output = Image.open(img_io)

    return output

def main():
    if "user_prompt_history" not in st.session_state:
            st.session_state["user_prompt_history"]=[]
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"]=[]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"]=[]
    if "model_history" not in st.session_state:
        st.session_state['model_history']=[]
    if 'files' not in st.session_state:
        st.session_state['files']=[]

    st.header("Chatbot")

    toggle_label = (
            'Click to Generate Text'
            if st.session_state.get('gen_mode', False)
            else 'Click to Generate Images'
        )

    toggle_value = st.session_state.get('gen_mode', False)
    is_toggle = st.toggle(toggle_label, value=toggle_value, key='gen_mode')

    with st.sidebar:
        if not is_toggle:
            st.title('Customize your Text Generation')
            uploaded_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)

            if uploaded_files:
                with st.spinner('Processing files...'):
                    process_files(uploaded_files)
            
            model = st.selectbox(
                "Choose your preferred Large Language Model",
                ('Claude 3 Sonnet',
                'Claude 3 Haiku',
                'Claude 3 Opus')
            )

            if model == 'Claude 3 Sonnet':
                model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
                st.info(
                """
                Claude 3 Sonnet strikes the ideal balance between intelligence and speed—particularly for enterprise workloads. It offers maximum utility, and is engineered to be the dependable for scaled AI deployments.
                \n**Max tokens:** 200K

                \n**Languages:** English, Spanish, Japanese, and multiple other languages.

                \n**Supported use cases:** RAG or search & retrieval over vast amounts of knowledge, product recommendations, forecasting, targeted marketing, code generation, quality control, parse text from images.
                """)
            elif model == 'Claude 3 Haiku':
                model_id = 'anthropic.claude-3-haiku-20240307-v1:0'
                st.info(
                    """
                    Anthropic’s fastest, most compact model for near-instant responsiveness. It answers simple queries and requests with speed.
                    \n**Max tokens: 200K**

                    \n**Languages:** English, Spanish, Japanese, and multiple other languages.

                    \n**Supported use cases:** Quick and accurate support in live interactions, translations, content moderation, optimize logistics, inventory management, extract knowledge from unstructured data.
                    """
                )
            elif model == 'Claude 3 Opus':
                model_id = 'anthropic.claude-3-opus-20240229-v1:0'
                st.info(
                    """
                    Anthropic’s most powerful AI model, with top-level performance on highly complex tasks. It can navigate open-ended prompts and sight-unseen scenarios with remarkable fluency and human-like understanding.
                    \n**Max Tokens:** 200K

                    \n**Languages:** English, Spanish, Japanese, and multiple other languages.

                    \n**Supported Use Cases:** Task automation, interactive coding, research review, brainstorming and hypothesis generation, advanced analysis of charts & graphs, financials and market trends, forecasting.
                    """
                )

            max_tokens = st.slider('Max Tokens', 1, 4096, 1024)
            temperature = st.slider('Temperature', 0.0, 1.0, 1.0)
        else:
            st.title('Customize your Image Generation')

            model = st.selectbox(
                "Choose your preferred Image Generator",
                ('Stable Diffusion XL')
            )

            if model == 'Stable Diffusion XL':
                model_id = 'stability.stable-diffusion-xl-v1'
                st.info(
                    """"
                    The most advanced text-to-image model from Stability AI.

                    \n**Max tokens:** 77-token limit for prompts

                    \n**Languages:** English

                    \n**Supported use cases:** Advertising and marketing, media and entertainment, gaming and metaverse.
                    """
                )

    prompt = st.chat_input("Enter your questions here")

    if prompt:
        if not is_toggle:
            with st.spinner("Generating......"):
                if uploaded_files:
                    output=get_response(query=prompt, chat_history=st.session_state["chat_history"], llm=load_llm(model_id, max_tokens, temperature), context=True)
                else:
                    output=get_response(query=prompt, chat_history=st.session_state["chat_history"], llm=load_llm(model_id, max_tokens, temperature), context=False)
            st.session_state["chat_history"].append((prompt, output))
        else:
            with st.spinner('Generating...'):
                output = get_image_response(prompt, model_id)

        st.session_state["chat_answers_history"].append(output)
        st.session_state["user_prompt_history"].append(prompt)        
        st.session_state['model_history'].append(model)

    if st.session_state["chat_answers_history"]:
            for i, j, k in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"], st.session_state['model_history']):
                message1 = st.chat_message("user")
                message1.write(j)
                message2 = st.chat_message("assistant")
                message2.write(k + ' Assistant: ')
                if isinstance(i, str):
                    message2.write(i)
                else:
                    st.image(i)
                
if __name__ == "__main__":
    main()