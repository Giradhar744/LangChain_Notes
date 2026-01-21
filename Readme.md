# 🦜🔗 LangChain Complete Study Guide

A comprehensive reference guide for LangChain - covering all core concepts, components, and implementation patterns.

---

## 📚 Table of Contents

1. [Introduction to LangChain](#introduction-to-langchain)
2. [LLMs (Large Language Models)](#llms-large-language-models)
3. [Chat Models](#chat-models)
4. [Prompts](#prompts)
5. [Chains](#chains)
6. [Embedding Models](#embedding-models)
7. [RAG (Retrieval Augmented Generation)](#rag-retrieval-augmented-generation)
8. [Agents](#agents)
9. [Tools](#tools)
10. [Runnables (LCEL)](#runnables-lcel)
11. [Structured Output](#structured-output)
12. [Best Practices](#best-practices)

---

## Introduction to LangChain

### What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides:

- **Integration**: Connect external data sources to LLMs
- **Agency**: Enable LLMs to interact with their environment
- **Modularity**: Swap components easily
- **Chains**: String actions together

### Core Principles

```
┌─────────────────────────────────────────────┐
│           LangChain Framework               │
├─────────────────────────────────────────────┤
│  Components → Chains → Agents → Apps       │
│                                             │
│  • Models (LLMs, Chat, Embeddings)         │
│  • Prompts (Templates, Selectors)          │
│  • Memory (Conversation, Vector)           │
│  • Indexes (Document Loaders, Splitters)   │
│  • Tools (APIs, Calculators, Search)       │
└─────────────────────────────────────────────┘
```

### Installation

```bash
pip install langchain
pip install langchain-openai
pip install langchain-community
```

---

## LLMs (Large Language Models)

### Overview

LLMs are text-completion models that take a string prompt and return a string completion.

### Key Characteristics

- **Input**: Raw text string
- **Output**: Text string
- **Use Case**: Simple text generation, completion tasks
- **Examples**: GPT-3, text-davinci-003

### Basic Usage

```python
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI(
    model_name="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_tokens=256
)

# Generate text
response = llm.invoke("Write a haiku about AI")
print(response)
```

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `temperature` | Randomness of output | 0.7 | 0.0 - 2.0 |
| `max_tokens` | Maximum response length | 256 | 1 - 4096+ |
| `top_p` | Nucleus sampling | 1.0 | 0.0 - 1.0 |
| `frequency_penalty` | Reduce repetition | 0.0 | -2.0 - 2.0 |

### LLM Flow Diagram

```
┌──────────────┐
│ Input Prompt │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     LLM      │
│  (OpenAI)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Response   │
│   (String)   │
└──────────────┘
```

---

## Chat Models

### Overview

Chat models are designed for conversational interfaces and work with message objects instead of raw text.

### Message Types

```python
from langchain_core.messages import (
    HumanMessage,      # User input
    AIMessage,         # AI response
    SystemMessage,     # System instructions
    FunctionMessage,   # Function/tool results
)
```

### Basic Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Chat Model
chat = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)

# Create messages
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is LangChain?")
]

# Get response
response = chat.invoke(messages)
print(response.content)
```

### Chat vs LLM

```
LLM Model:
Input:  "Write a poem about AI"
Output: "In circuits deep..."

Chat Model:
Input:  [SystemMessage("You are a poet"),
         HumanMessage("Write about AI")]
Output: AIMessage(content="In circuits deep...")
```

### Conversation Flow

```
┌─────────────────┐
│ System Message  │ ← Sets behavior/role
└────────┬────────┘
         │
┌────────▼────────┐
│ Human Message   │ ← User input
└────────┬────────┘
         │
┌────────▼────────┐
│   Chat Model    │
└────────┬────────┘
         │
┌────────▼────────┐
│  AI Message     │ ← Model response
└─────────────────┘
```

---

## Prompts

### Prompt Templates

Prompt templates help create reusable, parameterized prompts.

```python
from langchain.prompts import PromptTemplate

# Basic template
template = PromptTemplate(
    input_variables=["topic", "style"],
    template="Write a {style} article about {topic}"
)

# Format prompt
prompt = template.format(topic="AI", style="technical")
```

### Chat Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

# Create chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant"),
    ("human", "Help me with {task}"),
])

# Format
messages = chat_template.format_messages(
    role="coding",
    task="debugging Python"
)
```

### Few-Shot Prompting

```python
from langchain.prompts import FewShotPromptTemplate

# Define examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

# Create example template
example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Create few-shot template
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the opposite of each word:",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"]
)
```

### Prompt Engineering Best Practices

```
┌───────────────────────────────────────┐
│      Effective Prompt Structure       │
├───────────────────────────────────────┤
│                                       │
│ 1. Context/Role                      │
│    "You are an expert in..."         │
│                                       │
│ 2. Instructions                       │
│    "Analyze the following..."        │
│                                       │
│ 3. Examples (Few-shot)               │
│    "Example 1: Input → Output"       │
│                                       │
│ 4. Input Data                         │
│    {user_input}                      │
│                                       │
│ 5. Output Format                      │
│    "Respond in JSON format..."       │
│                                       │
└───────────────────────────────────────┘
```

---

## Chains

### Overview

Chains combine multiple components (LLMs, prompts, parsers) into a single pipeline.

### Simple Chain

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Create components
llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.run("eco-friendly water bottles")
```

### Sequential Chains

```python
from langchain.chains import SimpleSequentialChain

# Chain 1: Generate synopsis
chain_one = LLMChain(llm=llm, prompt=synopsis_prompt)

# Chain 2: Write review
chain_two = LLMChain(llm=llm, prompt=review_prompt)

# Combine chains
overall_chain = SimpleSequentialChain(
    chains=[chain_one, chain_two],
    verbose=True
)

# Run
review = overall_chain.run("a sci-fi novel about AI")
```

### Chain Types

```
┌────────────────────────────────────────┐
│           Chain Types                  │
├────────────────────────────────────────┤
│                                        │
│  LLMChain                             │
│  ├─ Basic: Prompt → LLM → Output     │
│  └─ Use: Simple transformations       │
│                                        │
│  SimpleSequentialChain                │
│  ├─ Linear: Chain1 → Chain2 → ...    │
│  └─ Use: Step-by-step processing     │
│                                        │
│  SequentialChain                      │
│  ├─ Multiple I/O between chains       │
│  └─ Use: Complex multi-step tasks    │
│                                        │
│  TransformChain                       │
│  ├─ Custom transformation logic       │
│  └─ Use: Data preprocessing           │
│                                        │
└────────────────────────────────────────┘
```

### Chain Flow

```
Input
  │
  ▼
┌─────────────┐
│   Prompt    │
│  Template   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     LLM     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Parser    │
└──────┬──────┘
       │
       ▼
    Output
```

---

## Embedding Models

### Overview

Embeddings convert text into numerical vectors that capture semantic meaning.

### Basic Usage

```python
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Embed single text
text = "This is a sample sentence"
vector = embeddings.embed_query(text)

# Embed multiple documents
docs = ["Text 1", "Text 2", "Text 3"]
vectors = embeddings.embed_documents(docs)
```

### Vector Similarity

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compare texts
text1_vec = embeddings.embed_query("I love programming")
text2_vec = embeddings.embed_query("I enjoy coding")
text3_vec = embeddings.embed_query("I like pizza")

similarity_1_2 = cosine_similarity(text1_vec, text2_vec)  # High
similarity_1_3 = cosine_similarity(text1_vec, text3_vec)  # Low
```

### Vector Space Visualization

```
         Programming
              │
    "I love coding" ●
              │     ╲
              │      ╲
    "I enjoy programming" ●
              │           ╲
              │            ╲
    ──────────┼──────────── ● "I like pizza"
              │              (Distant)
              │
         Technology
```

### Embedding Providers

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| OpenAI | text-embedding-ada-002 | 1536 | Low |
| OpenAI | text-embedding-3-small | 1536 | Lower |
| OpenAI | text-embedding-3-large | 3072 | Medium |
| HuggingFace | sentence-transformers | 384-768 | Free |
| Cohere | embed-english-v3.0 | 1024 | Low |

---

## RAG (Retrieval Augmented Generation)

### Overview

RAG combines information retrieval with text generation to provide accurate, context-aware responses.

### RAG Architecture

```
┌──────────────────────────────────────────────────┐
│                  RAG System                      │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────┐         ┌──────────────┐       │
│  │  Document  │──Split─→│    Chunks    │       │
│  │   Store    │         │              │       │
│  └────────────┘         └──────┬───────┘       │
│                                 │               │
│                          Embed  │               │
│                                 ▼               │
│                         ┌──────────────┐       │
│                         │    Vector    │       │
│      User Query ────────→   Database   │       │
│           │             └──────┬───────┘       │
│           │                    │               │
│           │              Retrieve Similar      │
│           │                    │               │
│           └────────┬───────────┘               │
│                    ▼                            │
│            ┌──────────────┐                    │
│            │  LLM with    │                    │
│            │   Context    │                    │
│            └──────┬───────┘                    │
│                   │                             │
│                   ▼                             │
│              Response                           │
│                                                  │
└──────────────────────────────────────────────────┘
```

### Complete RAG Implementation

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 5. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 6. Create QA chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. Query
query = "What is the main topic of the document?"
result = qa_chain({"query": query})
print(result["result"])
```

### Text Splitting Strategies

```python
# Character-based splitting
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

# Recursive splitting (recommended)
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)

# Token-based splitting
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
```

### RAG Chain Types

| Chain Type | Description | Use Case |
|------------|-------------|----------|
| `stuff` | Put all docs in single prompt | Small document sets |
| `map_reduce` | Process docs separately, combine | Large document sets |
| `refine` | Iteratively refine answer | Complex reasoning |
| `map_rerank` | Score and select best answer | Multiple sources |

### RAG Best Practices

```
┌─────────────────────────────────────┐
│       RAG Optimization Tips         │
├─────────────────────────────────────┤
│                                     │
│  1. Chunk Size                     │
│     • 500-1000 chars optimal       │
│     • Depends on content type      │
│                                     │
│  2. Overlap                         │
│     • 10-20% of chunk size         │
│     • Maintains context            │
│                                     │
│  3. Retrieval                       │
│     • k=3-5 documents              │
│     • Use MMR for diversity        │
│                                     │
│  4. Metadata                        │
│     • Add source, date, type       │
│     • Enable filtering             │
│                                     │
│  5. Re-ranking                      │
│     • Cross-encoder models         │
│     • Improve relevance            │
│                                     │
└─────────────────────────────────────┘
```

---

## Agents

### Overview

Agents use LLMs to decide which actions to take and in what order. They combine reasoning with tool usage.

### Agent Architecture

```
┌────────────────────────────────────────────┐
│              Agent System                  │
├────────────────────────────────────────────┤
│                                            │
│  User Query                                │
│      │                                     │
│      ▼                                     │
│  ┌─────────┐                              │
│  │  Agent  │◄─────┐                       │
│  │  (LLM)  │      │                       │
│  └────┬────┘      │                       │
│       │           │                       │
│       │ Decide    │ Observation           │
│       │ Action    │                       │
│       ▼           │                       │
│  ┌─────────┐     │                       │
│  │  Tools  │─────┘                       │
│  │ - Search│                              │
│  │ - Calc  │                              │
│  │ - API   │                              │
│  └─────────┘                              │
│       │                                    │
│       ▼                                    │
│  Final Answer                              │
│                                            │
└────────────────────────────────────────────┘
```

### Basic Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain import hub

# Define tools
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information"""
    # Implementation
    return f"Wikipedia results for: {query}"

def calculator(expression: str) -> str:
    """Perform calculations"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Search Wikipedia for factual information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Perform mathematical calculations"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Run agent
result = agent_executor.invoke({
    "input": "What is the population of Tokyo times 2?"
})
```

### Agent Types

```
┌────────────────────────────────────────┐
│         Agent Types                    │
├────────────────────────────────────────┤
│                                        │
│  ReAct Agent                          │
│  ├─ Reason and Act in steps          │
│  └─ Most versatile                   │
│                                        │
│  Conversational Agent                 │
│  ├─ Has conversation memory          │
│  └─ Good for chatbots                │
│                                        │
│  OpenAI Functions Agent               │
│  ├─ Uses OpenAI function calling     │
│  └─ More structured                  │
│                                        │
│  Self-Ask Agent                       │
│  ├─ Breaks down complex questions    │
│  └─ Good for research                │
│                                        │
│  Plan-and-Execute Agent               │
│  ├─ Creates plan first               │
│  └─ Executes step by step            │
│                                        │
└────────────────────────────────────────┘
```

### ReAct Pattern

```
Question: What is the capital of the country where the Eiffel Tower is located?

Thought: I need to find out where the Eiffel Tower is located first.
Action: Wikipedia
Action Input: "Eiffel Tower location"
Observation: The Eiffel Tower is in Paris, France.

Thought: Now I know it's in France. I should confirm the capital of France.
Action: Wikipedia
Action Input: "Capital of France"
Observation: The capital of France is Paris.

Thought: I now know the final answer.
Final Answer: Paris
```

### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create agent with memory
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Conversation
agent_executor.invoke({"input": "My name is Alice"})
agent_executor.invoke({"input": "What's my name?"})  # Remembers "Alice"
```

---

## Tools

### Overview

Tools are functions that agents can use to interact with external systems and perform specific tasks.

### Built-in Tools

```python
from langchain.agents import load_tools

# Load common tools
tools = load_tools(
    ["wikipedia", "llm-math", "requests_all"],
    llm=llm
)
```

### Custom Tools

```python
from langchain.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

# Method 1: Using @tool decorator
from langchain.tools import tool

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# Method 2: Using StructuredTool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name="Multiply",
    description="Multiply two numbers together"
)

# Method 3: Creating a custom class
class CustomSearchTool(BaseTool):
    name = "custom_search"
    description = "Useful for searching custom database"
    
    def _run(self, query: str) -> str:
        # Implementation
        return f"Results for: {query}"
    
    async def _arun(self, query: str) -> str:
        # Async implementation
        raise NotImplementedError("Async not implemented")
```

### Tool with Type Hints

```python
from typing import Optional
from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: Optional[int] = Field(
        default=5,
        description="Maximum number of results"
    )

@tool(args_schema=SearchInput)
def search_database(query: str, max_results: int = 5) -> str:
    """Search the database with specified parameters"""
    return f"Found {max_results} results for: {query}"
```

### Popular Tool Categories

```
┌────────────────────────────────────────┐
│         Tool Categories                │
├────────────────────────────────────────┤
│                                        │
│  Information Retrieval                │
│  • Wikipedia                          │
│  • Web Search (Google, Bing)         │
│  • News APIs                          │
│                                        │
│  Computation                           │
│  • Calculator                         │
│  • Python REPL                        │
│  • WolframAlpha                       │
│                                        │
│  Communication                         │
│  • Email                              │
│  • Slack                              │
│  • Twilio (SMS)                       │
│                                        │
│  Data Access                           │
│  • SQL Database                       │
│  • API Requests                       │
│  • File System                        │
│                                        │
│  Utilities                             │
│  • Web Scraping                       │
│  • Date/Time                          │
│  • Weather                            │
│                                        │
└────────────────────────────────────────┘
```

---

## Runnables (LCEL)

### LangChain Expression Language (LCEL)

LCEL provides a declarative way to compose chains with better streaming, async support, and observability.

### Basic Runnable

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Create components
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

# Chain with LCEL
chain = prompt | model | output_parser

# Invoke
result = chain.invoke({"topic": "programming"})
```

### Runnable Operations

```python
# Parallel execution
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    joke=prompt | model | output_parser,
    poem=poem_prompt | model | output_parser
)

# Branching
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x, code_chain),
    (lambda x: "math" in x, math_chain),
    default_chain
)

# Fallback
chain = primary_model.with_fallbacks([backup_model])

# Retry
chain = model.with_retry(stop_after_attempt=3)
```

### LCEL Advantages

```
Traditional Chain:
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Prompt  │──→│  Model  │──→│ Parser  │
└─────────┘   └─────────┘   └─────────┘
  Blocking      Blocking      Blocking

LCEL Chain:
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Prompt  │━━→│  Model  │━━→│ Parser  │
└─────────┘   └─────────┘   └─────────┘
  Streaming     Streaming     Streaming
  Async         Async         Async
  Parallel      Parallel      Parallel
```

### Streaming with LCEL

```python
# Stream tokens as they're generated
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### Complex LCEL Chain

```python
from operator import itemgetter

# Multi-step chain with branching
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language")
    }
    | RunnablePassthrough.assign(
        answer=lambda x: f"Context: {x['context']}\nQuestion: {x['question']}"
    )
    | prompt
    | model
    | output_parser
)

result = chain.invoke({
    "question": "What is LangChain?",
    "language": "English"
})
```

### Runnable Interface

```python
# All runnables support these methods:
runnable.invoke(input)           # Synchronous
runnable.batch([input1, input2]) # Batch processing
runnable.stream(input)           # Streaming
runnable.ainvoke(input)          # Async
runnable.abatch([input1, input2])# Async batch
runnable.astream(input)          # Async stream
```

---

## Structured Output

### Overview

Parse LLM outputs into structured formats like JSON, Pydantic models, or specific schemas.

### JSON Output

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Define schema
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")

# Create parser
parser = JsonOutputParser(pydantic_object=Person)

# Create prompt
prompt = PromptTemplate(
    template="Extract person information.\n{format_instructions}\n{query}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Chain
chain = prompt | model | parser

# Use
result = chain.invoke({"query": "John is a 30-year-old software engineer"})
# Output: {"name": "John", "age": 30, "occupation": "software engineer"}
```

### Pydantic Output Parser

```python
from langchain.output_parsers import PydanticOutputParser
from typing import List

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    pros: List[str] = Field(description="Positive aspects")
    cons: List[str] = Field(description="Negative aspects")
    summary: str = Field(description="Brief summary")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = PromptTemplate