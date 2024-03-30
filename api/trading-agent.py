import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json
import time
from dotenv import dotenv_values

llms = {}

@asynccontextmanager
async def lifespan(app: FastAPI):    
    # Make sure the model path is correct for your system!
    llms["llama"] = LlamaCpp(
        model_path="model/llama-2-13b-chat.Q5_K_M.gguf",
        n_ctx=4092, # context of token size
        n_gpu_layers=-1, #setting -1 to offload all the LLM layers to all available gpu 
        n_batch=4096, # no. of token in a prompt fed in LLM each time in a batch
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        max_tokens=2048 # max tokens LLM could generate/answer
    )
    yield  

app = FastAPI(lifespan=lifespan)

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

def process_command(llm, command):
    # Put user command into prompt (in future projects we'll be re-injecting whole chat history here)
    prompt = prompt_template.format("User: " + command)
    
    # Send command to the model
    # Assuming `llm.invoke` is a placeholder for sending command to an LLM (Language Learning Model)
    # and getting a response back
    output = llm.invoke(prompt, stop=["User:"])
    response = output
    
    # Yielding the initial response
    for token in response.split():
        yield token
    
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
            for token in final_response.split():
                yield token
    except Exception as e:
        print(e)
        final_response = str({"response": "The action is not triggered. Please try again."})
        for token in final_response.split():
            yield token


def run_llm(question: str) -> AsyncGenerator:
    llm : LlamaCpp = llms["llama"]
    response_iter = process_command(llm, question)
    for response in response_iter:
        time.sleep(0.3)
        yield f"response\n\n"

@app.get("/")
async def root(question: str) -> StreamingResponse:
    return StreamingResponse(run_llm(question), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
