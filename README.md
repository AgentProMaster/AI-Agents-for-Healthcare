# AI Agents for Healthcare A Nutritionist AI Agent Tutorial on Development and Publishing to AI Agent Marketplace
![AI Agent Marketplace](https://github.com/AgentProMaster/AI-Agents-for-Healthcare/blob/main/ai_agent_marketplace_screenshot.jpg)

## Introduction

In this tutorial, I will give you a detailed introduction on how to develop an AI Healthcare Agent and step by step methods to develop, deploy and publish your AI Agent to an AI Agent Marketplace (http://www.deepnlp.org/search/agent/) and make your AI Agents indexed by web search engines and visible to public. 

For example, we will develop an **Nutritionist AI Agent** for users who wants to **keep healthy and fit**, knowing exactly how many calories they take from their daily food and drinks, such as 100g of Starbucks coffees, etc. For the RAG (queries realtime food calories information), we are getting the nutrition information from trustworthy data sources, for example **United States Department of Agriculture** (USDA), etc.


## Prerequisite
- python 3.x
- pip package "HealthcareAgent", realtime API of health and medical data
- AI Agent Marketplace Account: We are using DeepNLP AI Agent Marketplace and Search Portal (http://deepnlp.org/search/agent/)


## Table of Content
1. Develop an Nutritionist AI Agent for Healthcare
2. Publish to AI Agent Marketplace
3. Alternative Sources of AI Healthcare Agent


### 1. Develop an Nutritionist AI Agent for Healthcare

**1.1 Nutritionist AI Agent Use Case Workflow**

```
User: What's the calories of starbucks coffee
Assistant: Hi, what type of coffee and how large is the cup?
User: I want to know the calories of "Double Shot Energy" coffee. The 15 Fluid Ounce bottle.
Assistant: Okay, Getting realtime data from data sources: www.usda.gov ....
Thinking...
Assistant: For a typical 15 Fluid Ounce Can Starbucks coffee, the energy is 47 kcal for every 100 ml. The 15 Ounce is roughly 15 ounce =
425.2 grams. The total calories are 200 kcal for each can of Starbucks Double Shot Energy Coffee Beverage.
```

To fetch realtime data from USDA website, we need to get the detailed information of products samples "under" brand "Starbucks", adn get the nutritions information from the USDA website. For example, goods name: Starbucks Double Shot Energy Coffee Beverage 15 Fluid Ounce Can
Brand Owner: Pepsico Inc. Brand: Starbucks 
https://fdc.nal.usda.gov/food-details/1167263/nutrients

![USDA Food Calories](https://github.com/AgentProMaster/AI-Agents-for-Healthcare/blob/main/usda_demo.jpg
)


**1.2 Install Package HealthcareAgent**


For quick implementation of RAG APIs fetching data, we are using the wrapped open API from pip package HealthcareAgent which can help us we query the brands and calories information from usda.gov website


```
pip install HealthcareAgent
```


**1.3 Running Examples**

```
import HealthcareAgent as ha

print ("DEBUG: ### Food Nutritions for Brand Starbucks")

result_dict = ha.api(api_name="food_calories_usda", query="Starbucks", topk=10)

item_list = result_dict["item_list"]
item_dict = result_dict["item_dict"]
item_list_pretty = result_dict["item_list_pretty"]

print (item_list)
print (item_dict)
[print (line) for line in item_list_pretty]

```


Sample Output


```

1. CHILLED COFFEE DRINK
$$starbucks CHILLED COFFEE DRINK(#image)$$
Every 100g:
Energy71.0KCALFiber, total dietary0.0G
Calcium, Ca85.0MG
Iron, Fe0.09MG
Vitamin A, IU0.0IU
Vitamin C, total ascorbic acid0.0MG
Protein2.19G
Total lipid (fat)1.09G
Carbohydrate, by difference12.9G
Total Sugars11.2G
Sodium, Na36.0MG
Cholesterol5.0MG
Fatty acids, total trans0.0G
Fatty acids, total saturated0.73G


2. CLASSIC HOT COCOA MIX, CLASSIC
$$starbucks CLASSIC HOT COCOA MIX, CLASSIC(#image)$$
Every 100g:
Energy357KCALProtein7.14G
Total lipid (fat)8.93G
Carbohydrate, by difference78.6G
Total Sugars60.7G
Fiber, total dietary10.7G
Calcium, Ca57.0MG
Iron, Fe14.3MG
Potassium, K986MG
Sodium, Na0.0MG
Vitamin D (D2 + D3), International Units0.0IU
Sugars, added57.1G
Cholesterol0.0MG
Fatty acids, total trans0.0G
Fatty acids, total saturated5.36G



```


**1.4 LLM Tool based OpenAI Function Calling**

Once we have the API and user prompt e.g. "What is the calories of a can of Starbucks Double Shot Energy Coffee Beverage? ", we need to define a python function and ask LLM (OpenAI ChatGPT) to help us define the brand names from user query and summarize the results of calorie information to calulcate the exact calories of different size and weight (e.g. cup size tall, grande, etc. 100 ml, ounces, etc.) of coffee brands (black coffee, frappuccino, latte).


The workflow is like:

user query -> define function fetch_calories_usda() -> Calling LLM to Fill Parameters -> Execution of the Code -> RAG: Execution to Get Food Calories Information

-> Reasoning: Calculate the exact product's calories based on RAG results

-> Response Generation: Generate text response


```

import os
import HealthcareAgent as ha

import agentboard as ab
from agentboard.utils import function_to_schema


prompt = "What is the calories of a can of Starbucks Double Shot Energy Coffee Beverage? "

def fetch_calories_usda(brand_name: str, product_name: str):
    
    result_dict = ha.api(api_name="food_calories_usda", query=brand_name, topk=50)

    return result_dict


def run_example():

    from openai import OpenAI
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    if OPENAI_API_KEY is None:
        OPENAI_API_KEY = "your_openai_key"
        print ("DEBUG: Please Input Your OpenAI key")
        return
    openai.api_key = OPENAI_API_KEY

    ## 1. Fetching Data: calling OpenAI
    tools = [fetch_calories_usda]
    tools_map = {tool.__name__:tool for tool in tools}
    tools_schema = [function_to_schema(tool) for tool in tools]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather like in Paris today?"}],
        tools=tools_schema,
    )

    print(completion.choices[0].message.tool_calls)

    ## execution the function and get the result
    result_list = []
    for i, tool_call in enumerate(tool_calls):
        ## kwargs json dict
        arguments = json.loads(tool_call['function']['arguments'])
        function_name = tool_call['function']['name']
        cur_tool = tools_map[function_name] if function_name in tools_map else None

        ## add log, logs will grouped by the same process_id and displayed
        ab.summary.dic(name="Function excecution input params", data=[arguments], process_id="tool_execution_%s" % i)
        ab.summary.tool(name="Function excecution function name %s" % cur_tool.__name__, data=[cur_tool], process_id="tool_execution_%s" % i)

        result = cur_tool(**arguments) if func is not None else None
        print (result)
        result_list.append(result)  

    ## 2. Cacluate the Calories 
    ## 3. Generating Text Response



```





## 2. Publish AI Agent to AI Agent Marketplace

![AI Agent Marketplace](https://github.com/AgentProMaster/AI-Agents-for-Healthcare/blob/main/ai_agent_marketplace_screenshot.jpg)

**2.1 Wrap up a Restful API of your AI Agent on Web Server**


Once you have your API level function of AI Agent ready, You can deploy your Healthcare AI Agent to python website and start a RESTFUL API and serving.

***Prerequisite***
Linux Server
Flask WebService Framwork

The detailed process are omitted from this tutorial and to deploy a RESTFUL API taking user input prompt and respond with the answers, you can find plenty of resources on the website,
such as Flask web service, etc.


Let's say your published AI Agent Post Request Address is "http://{your_domain}:{port}/post", and you can publish your AI Agent and start indexed on Google Search Engine!


**2.2 Register and Publish Your AI Agent on AI Agent Marketplace**

First of all, you can register your AI Agent on AI Agent Marketplace (http://www.deepnlp.org/search/agent/). There are already many different types of AI agents  (Autonomous Agents, Finance Agents, etc),
for large corporations (Microsoft AI Agent/Claude AI Agent/Google AI Agent/Salesforce AI Agent) as well as individual developers.

Click on the button "Register" Register Your First AI App Agent Listing for Free, and you will be directed to a login page if you are not loggedin or registered yet. 

Click "Workspace" -> "AI Services" on the banner or Visit Directly http://www.deepnlp.org/workspace/my_ai_services

The schema of registering your AI Agents contains below fields:


| Field | Description | Example|
|-----|-----|-----| 
|Publisher | Your Company Name or Individual Developers | pub-my-healthcare-agent |
|Content Name |Name of the AI Agent | Nutritionist AI Agent |
|Website | Website Address |  http://www.example.com  |
|Content | Description of the function of your AI Agents | A Nutritionist AI Agent which can help user calculate the calories of daily food. Useful for people on the diet or fitness enthusiastic. |
|Category | Choose Option of Category |  AI Agent  |
|Sub-Category | Choose Option of Category |  AI Agent  |
|Content Tag List | The tag describing the functions of your Agent | Healthcare,Fitness |
|Github | Your Github Address | https://github.com/AgentProMaster/AI-Agents-for-Healthcare |
|Price | Optional | Free |
|REST API DEMO | http://{your_domain}:{port}/post | http://www.example.com:8080/post |


The Rest API Demo will be called when user clicked the Try API button on the publisher detailed pages.


**2.3 Publisher your AI Agent**

After you published your AI Agent, the status will be pending and once approved you can visit the webpage of your published AI Agent at the address at 

AI Agent Webpage Sample: 

http://www.deepnlp.org/store/ai-agent/ai-agent/pub-salesforce-ai-agent/salesforce-agentforce-ai-agent-review

Your AI Agent will the be indexed by AI Agent Marketplace Search portal (http://www.deepnlp.org/search/agent/) and the detailed AI Agent page will also
be indexed by Google within a short period of time (Depending on Google Crawler frequency and the content of your AI Agent). 



## 3. Alternative Sources of AI Healthcare Agent

**3.1  USDA FoodData Central**
Offers calorie counts, nutrient details, and a nutrient density rating for various foods.
Website: nutritiondata.self.com
Government and Academic Resources
USDA FoodData Central

The official database from the U.S. Department of Agriculture. It provides accurate and verified calorie counts for raw, processed, and prepared foods.
Website: http://fdc.nal.usda.gov


**3.2 Canada's Food Guide Nutritional Facts**
A resource from the Canadian government that provides calorie and nutrient information for foods common in Canada.
Website: www.food-guide.canada.ca
Apps and Online Tools



## 4. Related Reading
### AI Agent Blogs
[AI Agent Frameworks Benchmarks Types Examples and Marketplace Review A Comprehensive List](http://www.deepnlp.org/blog/ai-agent-review-benchmarks-and-environment-a-comprehensive-list) <br>
[AgentBoard: AI Agent Visualization Toolkit for Agent Loop Workflow RAG Tool Use Functions Callings and Multi Modal Data Visualization](http://www.deepnlp.org/blog/agentBoard-ai-agent-visualization-toolkit-agent-loop-workflow) <br>
[AI Agent Resources](http://www.deepnlp.org/blog?category=agent) <br>
[Introduction to multimodal generative models](http://www.deepnlp.org/blog/introduction-to-multimodal-generative-models) <br>
[Generative AI Search Engine Optimization](http://www.deepnlp.org/blog/generative-ai-search-engine-optimization-how-to-improve-your-content) <br>
[AI Image Generator User Reviews](http://www.deepnlp.org/store/image-generator) <br>
[AI Video Generator User Reviews](http://www.deepnlp.org/store/video-generator) <br>
[AI Chatbot & Assistant Reviews](http://www.deepnlp.org/store/chatbot-assistant) <br>
[AI Store-Best AI Tools User Reviews](http://www.deepnlp.org/store/pub/) <br>
[AI Store Use Cases-Best AI Tools Cases User Reviews](http://www.deepnlp.org/store) <br>

#### Search AI Agent & Robotics
[AI Agent Search](http://www.deepnlp.org/search/agent/) <br>
[Robot Search](http://www.deepnlp.org/search/robot) <br>
[AI & Robot Comprehensive Search](http://www.deepnlp.org/search) <br>

#### AI Agent Review
[Microsoft AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-microsoft-ai-agent) <br>
[Claude AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-claude-ai-agent) <br>
[OpenAI AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-openai-ai-agent) <br>
[AgentGPT AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-agentgpt) <br>
[Saleforce AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-salesforce-ai-agent) <br>
[Google AI Agents Reviews](http://www.deepnlp.org/store/pub/pub-google-ai-agent) <br>

#### AI Agent Workflow and AgentBoard
[AI Agent Board Visualization](https://ai-hub-admin.github.io/agentboard) <br>
[AI Agent Board](https://ai-hub-admin.github.io/agentboard/) <br>
[AI Agent Pro Master](https://agentpromaster.github.io/ai-agent/) <br>
[RAG Visualizer Agent](https://aiforce1024.github.io/rag_tutorial/) <br>
[Chat Visualizer Agent](https://llmpro70b.github.io/Chat-Visualizer/) <br>
[AI Agent Marketplace](https://agentpromaster.github.io/AI-Agent-Marketplace/) <br>

#### Text2Video Generation
[Sora User Review](http://www.deepnlp.org/store/pub/pub-sora) <br>
[Sora vs Kling AI Review](http://www.deepnlp.org/store/compare/pub-sora-vs-pub-kling-kwai) <br>
[Sora vs Dreamina Review](http://www.deepnlp.org/store/compare/pub-sora-vs-pub-dreamina-douyin) <br>
[Sora vs Runway Review](http://www.deepnlp.org/store/compare/pub-sora-vs-pub-runway) <br>
[Sora vs Lumina Review](http://www.deepnlp.org/store/compare/pub-sora-vs-pub-luma-ai) <br>
