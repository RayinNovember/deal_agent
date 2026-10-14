from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
#from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from importlib.metadata import version

print(f'langchain-core version: {version("langchain-core")}')
print(f'langgraph version: {version("langgraph")}')

def main():
    # testing OpenAI
    llm = ChatOpenAI(model_name = 'gpt-4o-mini',temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")
    
    # testing Anthropic
    llm_anthropic = ChatAnthropic(model = 'claude-sonnet-4-5-20250929', temperature=0)
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")
    
    print('Setup Complete!')
    
if __name__ == '__main__':
    main()