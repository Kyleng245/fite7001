import streamlit as st
from dotenv import load_dotenv
import os
from htmlTemplate import css, bot_template, user_template
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
# from langchain.vectorstores import FAISS
# Use Langchain prompt messages, e.g., System Message, Human Message to provide context and instruct how LLM could answer our questions
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
# We use the Question-answer chain where it has arguments accepting chat history and documents (i.e., the python code in this case)
from langchain.chains.question_answering import load_qa_chain
# use the langchain document parser to parse the trading code
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import numpy as np
# This is a script to calculate Moving Average Convergence Divergence trading strategy
import vectorbt as vbt
import yfinance as yf
import io
import sys
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json
from dotenv import dotenv_values
import time
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pyperclip

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

load_dotenv()

llm = LlamaCpp(
        model_path="../model/llama-2-13b-chat.Q5_K_M.gguf",
        temperature=0.75,
        max_tokens=2048,
        top_p=1,
        n_ctx=3000)

# llm = LlamaCpp(
#     model_path="../model/llama-2-13b-chat.Q5_K_M.gguf",
#     n_ctx=8192, # context of token size
#     n_gpu_layers=-1, #setting -1 to offload all the LLM layers to all available gpu 
#     n_batch=4096, # no. of token in a prompt fed in LLM each time in a batch
#     f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
#     verbose=False,
#     max_tokens=2048 # max tokens LLM could generate/answer
# )


llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- If the document does not contain relevant information, state "I cannot provide an answer based on the provided document."
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}
[/INST]
"""

prompt_template = '''<s>[INST] <<SYS>>
Assistant is a expert JSON builder designed to assist with a wide range of tasks.

Assistant is able to trigger actions for User by responding with JSON strings that contain "action" and "action_input" parameters.

Actions available to Assistant are:

- "StockExecutionTool": Useful for when you need to execute stock trading orders for users.
  - To use the StockExecutionTool tool, Assistant should write like so:
    ```json
    {{"action": "StockExecutionTool",
      "action_input": "('symbol', 'side', 'quantity')"}}
    ```
  * `symbol` is the string of stock symbol for trading
  * `side` should be enum of string where "buy" to buy the stork or "sell" to sell the stock
  * `quantity` is the float number of quantity of stock to trade
- "ViewAccountTool": Useful for when you need to query the trading account status such as buying power, and Profit and Loss
  - To use the ViewAccountTool tool, Assistant should write like so:
    ```json
    {{"action": "ViewAccountTool",
      "action_input": 'view_action'}}
    * the view_action is an enum of string where it could be 'buying_power' if you want to view the buying power of the account or 'PnL' if you want to view the Profit or Loss
    ```
- "ViewPositionTool": Useful for when you need to query the position of individual stocks or all the stocks
  - To use the ViewAccountTool tool, Assistant should write like so:
    ```json
    {{"action": "ViewPositionTool",
      "action_input": 'symbol'}}
  * `symbol` is the string of stock symbol to view the position, in case the user want to query all the position, please return "ALL_STOCK" as symbol 
    ```

Here are some sample conversations between the Assistant and User:

User: Hey how are you today?
Assistant: ```json
{{"action": "Final Answer",
 "action_input": "I'm good thanks, how are you?"}}
```
User: I'm great, could you help buy 1 Apple Stock?
Assistant: ```json
{{"action": "StockExecutionTool",
 "action_input": "('AAPL', 'buy', '1')"}}
```
User: Could I view my buying power of my trading account?
Assistant: ```json
{{"action": "ViewAccountTool",
 "action_input": "buying_power"}}
```

User: I want to know the profit and loss of my account
Assistant: ```json
{{"action": "ViewAccountTool",
 "action_input": "PnL"}}
```

You are only allowed to return ```json {{"action": ..., "action_input"}} ``` .

<</SYS>>

{0}[/INST]'''


config=dotenv_values("../.env")
trading_client = TradingClient(config['ALPACA_API_KEY'], config['ALPACA_SECRET_KEY'])

def get_text_chunks():
    # Load
    try:
        loader = GenericLoader.from_filesystem(
            "../code",
            glob="**/*",
            suffixes=[".py"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=0), # minimum lines of code to activate parsing, default=0, https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/parsers/language/language_parser.py
        )
        documents = loader.load()

        # Split the python code in recursive splitter
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/code_splitter
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=2500, # maximum of tokens that are parsed in a file, exceeding it will go to next texts
            chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(documents)

        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=512,
        #     chunk_overlap=256,
        # )
        # split_docs = text_splitter.create_documents(content, metadatas=metadata)
        print(f"Split documents into {len(split_docs)} passages")
        return split_docs   
    except Exception as e:
        print(e, 'text chunks')


def ingest_into_vectordb(split_docs):
    # shared use
   
    embeddings = HuggingFaceEmbeddings(cache_folder="../model", model_kwargs={"device":"cuda:0"})
    db = Chroma.from_documents(split_docs, embeddings)
    return db
   
    # we uses mmr as similarity search method and sets nearest neighbor as 1 to only extract one trading strategy


def get_conversation_chain():
    # provide the system prompt to make the LLM aware of its mission to generate trading code
    template_messages = [
        SystemMessage(content="""
                    You are a helpful assistant in coding Python trading strategy. Please utilize the code in the code base.
                    When you write python code, please enclose your python block by ```python ...your python code ```.  
                    """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("Here is my codebase {context} \n Question: {text}"),
    ]

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, input_key="text")

    template = ChatPromptTemplate.from_messages(template_messages)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=template, memory=memory, verbose=True)

    print("Conversational Chain created for the LLM using the vector store")
    return chain

def handle_userinput(retriever, question):
    docs = retriever.get_relevant_documents(question)
    print(question, 'questionssdfsdfsd')
    print(docs, 'docs')
    response = st.session_state.conversation({"input_documents":docs, "text": question})
    print(response, 'response')
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        print(i, message, 'message')
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            print(message.content)
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# Function to display chatbot page
# def display_chatbot(retriever):
   
def execute_code(code):
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        # Execute the code
        exec(code, globals(), locals())
    except Exception as e:
        st.error(f"Error executing code: {e}")
        return

    # Restore standard output
    sys.stdout = old_stdout

    # Capture the output
    output_str = new_stdout.getvalue()

    return output_str

def display_code_terminal():
    st.title("Trading Strategy Code IDE")

    # Code editor
    code = st.text_area("Enter your Python code here:", height=300)

    st.write("Code Preview")
    st.code(code, language='python')
    

    # Button to execute code
    if st.button("Run"):
        # Remove lines starting with 'import'
        code_lines = code.split('\n')
        filtered_code_lines = [line for line in code_lines if not line.strip().startswith('import')]
        filtered_code = '\n'.join(filtered_code_lines)
        
        modified_code = filtered_code.replace("\\n", "\n").replace(" ", "").strip()
         # Remove the first and last \n characters
        if modified_code.startswith("\\n"):
            modified_code = modified_code[2:]
        if modified_code.endswith("\\n"):
            modified_code = modified_code[:-2]
        
        modified_code = execute_code(modified_code)

# Function to display trading agent page
def display_trading_agent():
    st.title("Trading Agent Page")
    if "trading_conversation" not in st.session_state:
        st.session_state.trading_conversation = None
    if "trading_chat_history" not in st.session_state:
        st.session_state.trading_chat_history = []
    user_question = st.text_input("I'm here to provide trading services. Feel free to ask any questions or make requests related to your paper trades.", placeholder='Message TradingAgent...')
    st.caption("The conversation history would not be stored due to privacy protection")

    if user_question:
        with st.spinner("Loading"):
            process_command(user_question)


def execute_order(symbol, side, quantity):
    """Execute stock trading orders based on the stock symbol, position side, and quantity"""

    # preparing market order
    market_order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        time_in_force=TimeInForce.DAY
                        )
    # Market order
    market_order = trading_client.submit_order(
                    order_data=market_order_data
                )
    return {"order_id": market_order.client_order_id, "order_status": market_order.status}

def view_account_status(info_type):
    account = trading_client.get_account()
    if info_type == 'buying_power':
        return {'response': f'${account.buying_power} is available as buying power.' }
    if info_type == 'PnL':
        balance_change = float(account.equity) - float(account.last_equity)
        return {'response': f'Today\'s portfolio balance change: ${balance_change}' }
    
def get_position(symbol):
    if symbol == "ALL_STOCK":
        portfolio = trading_client.get_all_positions()
        # Print the quantity of shares for each position.
        positions = {}
        for position in portfolio:
            position = json.loads(position.model_dump_json())
            positions[position['symbol']] = {key: position[key] for key in ['qty', 'side', 'cost_basis', 'market_value']}
        return {"response": positions}
    else:
        position = json.loads(trading_client.get_open_position(symbol).model_dump_json())
        selected_attributes = {key: position[key] for key in ['symbol', 'qty', 'side', 'cost_basis', 'market_value']}
        return {"response": selected_attributes}

def process_command(command):
    # llm = llms['LLama']
   
    # Put user command into prompt (in future projects we'll be re-injecting whole chat history here)
    prompt = prompt_template.format("User: " + command)
    
    # Send command to the model
    # Assuming `llm.invoke` is a placeholder for sending command to an LLM (Language Learning Model)
    # and getting a response back
    output = llm.invoke(prompt, stop=["User:"])
    response = output
    
    # # # Yielding the initial response
    # for token in response.split():
    #     yield token
    
    # Initialize result outside of try-except block
    result = None
    
    # Try to process the response and perform actions
    try:
        # Extract json from model response by finding first and last brackets {}
        firstBracketIndex = response.index("{")
        lastBracketIndex = len(response) - response[::-1].index("}")
        jsonString = response[firstBracketIndex:lastBracketIndex]
        responseJson = json.loads(jsonString)
        
        if responseJson['action'] == 'StockExecutionTool':
            action_input = eval(responseJson['action_input'])
            result = execute_order(action_input[0], action_input[1], float(action_input[2]))
        elif responseJson['action'] == 'ViewAccountTool':
            action_input = responseJson['action_input']
            result = view_account_status(action_input)
        elif responseJson['action'] == 'ViewPositionTool':
            action_input = responseJson['action_input']
            result = get_position(action_input)
            
        # If result was computed, yield it
        if result:
            final_response = str(result)
            st.session_state.trading_chat_history = final_response
            st.write(user_template.replace(
                        "{{MSG}}", command), unsafe_allow_html=True)
            st.write(bot_template.replace(
                        "{{MSG}}", st.session_state.trading_chat_history), unsafe_allow_html=True)
    except Exception as e:
        print(e)
        final_response = str({"response": "The action is not triggered. Please try again."})
        st.session_state.trading_chat_history = final_response
        st.write(user_template.replace(
                        "{{MSG}}", command), unsafe_allow_html=True)
        st.write(bot_template.replace(
                        "{{MSG}}", st.session_state.trading_chat_history), unsafe_allow_html=True)
 
# def copy_to_clipboard(text):
#     pyperclip.copy(text)
#     st.success(f"Command copied to clipboard!")

def copy_to_clipboard(text):
    st.write(text)
    

def main():
    load_dotenv()

    try:
        st.set_page_config(page_title="Finance GPT",
                        page_icon=":chart_with_upwards_trend:")
        st.write(css, unsafe_allow_html=True)

        st.sidebar.title("Navigation")

        page = st.sidebar.radio("Go to", ["Chatbot", "Code Terminal", "Trading Agent"])

        sidebar_items=[]

        # Display suggested commands based on the selected category
        if page == "Chatbot":
            st.sidebar.write('Suggested Prompts')
            sidebar_items = [ "How can I leverage Bollinger Bands (BB) trading strategy to trade Google Stocks?", "Can you provide code examples using Moving Average Convergence Divergence (MACD) to trade Google Stocks?", "How can I apply the Mean Reversion (MR) trading strategy to trade Google Stocks?", 
                         "How to use Relative Strength Index (RSI) trading strategy with oversold threshold as 20 and overbought threshold as 80 for trading in Microsoft Stocks?", "Can you provide code examples for implementing the Volatility Breakout (VB) strategy using default parameter values when trading Microsoft Stocks?",
                         "Can you list code scripts for implementing Simple Moving Average (SMA) trading strategy with short moving average window size as 20 and long moving average window size as 100 for trading in Microsoft Stocks?"
                         ]
        elif page == "Trading Agent":
            st.sidebar.write('Suggested Prompts')
            sidebar_items = ["Can you help buy me 10 Google Stock?", "what is my current account buying power. Thanks.", "I want to know my current PnL Thanks.", "I want to know my position of Apple Stock.", "I want to know my position of all my portfolio."]

        # Display each item with a circular border
        for item in sidebar_items:
            with st.sidebar:
                st.caption(f"<div style='border: 1px solid #ccc; border-radius: 4%; padding: 10px; margin: 5px;'>{item}</div>", unsafe_allow_html=True)

        if page == "Chatbot":
                st.title("Chatbot Page")

                if "conversation" not in st.session_state:
                    st.session_state.conversation = None
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                if "is_processed" not in st.session_state:
                    st.session_state.is_processed = False

                with st.spinner("Loading"):
                        # create conversation chain
                        # get the text chunks
                        if st.session_state.is_processed is False:
                            split_docs = get_text_chunks()

                                        # create vector store
                            vectorstore = ingest_into_vectordb(split_docs)

                            retriever = vectorstore.as_retriever(
                                search_type="mmr",  # Also test "similarity"
                                search_kwargs={"k": 1},
                                )
                            st.session_state.retriever = retriever
                            st.session_state.conversation = get_conversation_chain()
                            st.session_state.is_processed = True
                # If the conversation has already been initialized, retrieve the existing retriever
                retriever = st.session_state.retriever

                user_question = st.text_input("I'm here to offer trading strategies. How can I help you?", placeholder='Message FinanceGPT...')

                if user_question:
                    handle_userinput(retriever, user_question)        

        elif page == "Code Terminal":
            display_code_terminal()
        elif page == "Trading Agent":
            display_trading_agent()
    except Exception as e:
        print(e)
        st.error(e)
if __name__ == '__main__':
    main()
