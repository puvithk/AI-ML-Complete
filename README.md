# AI-ML-Complete: A Systematic Study of Transformers, LLMs, and Agentic Orchestration

Welcome to the **AI-ML-Complete** repository! This project documents a systematic, hands-on study journey of Artificial Intelligence and Machine Learning. While the first 30 days are complete, this is an ongoing learning process with more milestones to come. The study progresses step-by-step from first-principles mathematical implementations in NumPy, through standard deep learning architectures in PyTorch, to state-of-the-art agentic pipelines and persistent, resilient multi-agent orchestration using LangGraph, Smolagents, and LlamaIndex.

Each day represents a complete, runnable module exploring a fundamental concept, complete with scratch implementations, unit tests, notebooks, and real-world tools.

---

## 🗺️ Core Technical Roadmap

| Phase | Days | Core Focus | Key Technologies |
| :--- | :--- | :--- | :--- |
| **1. Transformer Foundations** | Days 1–9 | Mathematical models & architectures from scratch | Python, NumPy, Keras, TensorFlow, Hugging Face Tokenizers |
| **2. Deep Learning in PyTorch** | Days 10–12 | PyTorch mechanics and training sequence models | PyTorch, Hugging Face Datasets, CUDA |
| **3. Agentic & RAG Frameworks** | Days 13–20 | Retrieval pipelines, tools, and multi-agent coordination | Smolagents, LlamaIndex, ChromaDB, Google APIs |
| **4. Advanced LangGraph States** | Days 21–30 | Stateful agents, memory, validation, and persistent DBs | LangGraph, LangChain, SQLite, Pydantic |

---

## 📅 Day-by-Day Technical Summary

### 🧠 Phase 1: Transformer Foundations from Scratch (Days 1–9)

#### 🔹 Day 1: NumPy Scaled Dot-Product Self-Attention
* **Focus**: Implementing the core mathematical building block of modern LLMs.
* **Key Concept**: Programmed single-head self-attention using pure NumPy based on the formula:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **Files**: [selfAttention.py](file:///f:/AI%20ML/AI-ML-Complete/Day%201/selfAttention.py)

#### 🔹 Day 2: Multi-Head Attention in NumPy
* **Focus**: Scaling attention to process information from multiple subspaces in parallel.
* **Key Concept**: Programmed projection weights ($W_q$, $W_k$, $W_v$) for multiple heads, performed independent dot-product attention per head, concatenated outputs, and applied a final linear projection $W_o$ to map back to the embedding dimension.
* **Files**: [selfAttention.py](file:///f:/AI%20ML/AI-ML-Complete/Day%202%20Multi%20head/selfAttention.py)

#### 🔹 Day 3: Sinusoidal Positional Encodings
* **Focus**: Adding sequence order awareness into permutation-invariant attention layers.
* **Key Concept**: Implemented wave positional encodings using sine functions for even dimensions and cosine functions for odd dimensions across token indices:
  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
  $$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
* **Files**: [posistional.py](file:///f:/AI%20ML/AI-ML-Complete/Day%203%20Positioanl%20head/posistional.py)

#### 🔹 Day 4: Layer Normalization from Scratch
* **Focus**: Ensuring training stability and preventing gradient explosion/vanishing issues.
* **Key Concept**: Wrote custom NumPy Layer Normalization. Computes mean and variance across the feature dimension (axis -1) for each sample and normalizes activations to zero-mean and unit-variance with numerical epsilon stability:
  $$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
* **Files**: [layerNormalization.py](file:///f:/AI%20ML/AI-ML-Complete/Day%204%20layerNorm/layerNormalization.py)

#### 🔹 Day 5: Residual Connections & Skip Architectures
* **Focus**: Enabling deep network training through identity mapping.
* **Key Concept**: Built a complete ResNet convolutional block using TensorFlow/Keras. Explored the addition of skip connections (residual add) and implemented a $1 \times 1$ convolutional shortcut to align channel dimensions when shortcut and residual paths mismatch.
* **Files**: [resnet.py](file:///f:/AI%20ML/AI-ML-Complete/Day%205%20residual%20connections/resnet.py)

#### 🔹 Day 6: Scratch-Built NumPy Transformer Encoder Block
* **Focus**: Assembling the full feed-forward encoder pipeline from scratch.
* **Key Concept**: Tied previous components into a unified, modular `Encoder` class. The pipeline: Input Embeddings $\rightarrow$ Sinusoidal Encodings $\rightarrow$ Multi-Head Self-Attention (using `np.einsum`) $\rightarrow$ Add & Normalize $\rightarrow$ Feed Forward Network (2 linear layers + ReLU) $\rightarrow$ Add & Normalize.
* **Files**: [encoder.py](file:///f:/AI%20ML/AI-ML-Complete/Day%206%20Encoder/encoder.py)

#### 🔹 Day 7: Scratch-Built NumPy Transformer Decoder Block
* **Focus**: Creating autoregressive generating layers with causal masking.
* **Key Concept**: Implemented a complete `Decoder` class in pure NumPy. Features a Causal Attention Mask (upper-triangular matrix set to $-\infty$) in the self-attention layer to prevent tokens from attending to future tokens, followed by Cross-Attention (Queries from decoder, Keys/Values from encoder), and a Feed-Forward network.
* **Files**: [decoder.py](file:///f:/AI%20ML/AI-ML-Complete/Day%207%20Decoder/decoder.py)

#### 🔹 Day 8: Custom BPE Tokenizer Training
* **Focus**: Translating characters/words into discrete vocabulary tokens.
* **Key Concept**: Trained a custom BPE tokenizer using Hugging Face `tokenizers` on the Salesforce `wikitext-103-raw-v1` dataset. Configured special tokens (`[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`) and analyzed subword vocabularies.
* **Files**: [tokenizer.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%208%20Tokenizer/tokenizer.ipynb)

#### 🔹 Day 9: Cross-Entropy Loss from First Principles
* **Focus**: Implementing classification loss metrics with numerical stability.
* **Key Concept**: Programmed Binary Cross-Entropy Loss, Binary Cross-Entropy Cost, and Categorical Cross-Entropy in NumPy. Implemented epsilon clipping ($1e^{-15}$) on predicted probabilities to prevent undefined $\log(0)$ computational errors.
* **Files**: [cross_entropy.py](file:///f:/AI%20ML/AI-ML-Complete/Day%209%20Cross%20Entropy/cross_entropy.py)

---

### 🔥 Phase 2: Deep Learning in PyTorch (Days 10–12)

#### 🔹 Day 10: PyTorch Core & Module Subclassing
* **Focus**: Transitioning from scratch NumPy to dynamic PyTorch graph operations.
* **Key Concept**: Explored `nn.Sequential`, backpropagation, and optimizer steps. Subclassed `nn.Module` to build a custom `LinearRegression` model to predict sales using scaled advertising datasets with MSE Loss.
* **Files**: [neuralNetwork.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2010%20pytorch/neuralNetwork.py)

#### 🔹 Day 11: End-to-End Seq2Seq Transformer in PyTorch
* **Focus**: Re-implementing the full sequence-to-sequence Transformer in PyTorch.
* **Key Concept**: Designed custom PyTorch modules for MHSA (with query-key-value matrix splitting and combined transpositions), FFN, Embeddings, Positional Encodings, and unified them in a 4-layer Seq2Seq `Transformer` module. Programmed a dummy training loop using Adam and Cross-Entropy.
* **Files**: [transformer.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2011%20LLM%20using%20pytorch/transformer.py)

#### 🔹 Day 12: Training a PyTorch Transformer on Real Text
* **Focus**: Optimizing and deploying the PyTorch Transformer on actual text corpora.
* **Key Concept**: Loaded `fka/prompts.chat` using Hugging Face `datasets`. Wrote a custom word-level tokenizer with padding, loaded tensors in batches, trained the model with gradient clipping (`clip_grad_norm_`), and implemented an autoregressive greedy decoding `generate_text` helper.
* **Files**: [transformer.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2012/transformer.ipynb)

---

### 🛠️ Phase 3: Agentic Architectures & LlamaIndex (Days 13–20)

#### 🔹 Day 13: Introduction to Smolagents
* **Focus**: Developing LLM agents that write and execute code.
* **Key Concept**: Built a basic search agent using Hugging Face's `smolagents` library. Instantiated a `CodeAgent` armed with `DuckDuckGoSearchTool` and `InferenceClientModel`, allowing the agent to write and execute python snippets locally to solve queries.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2013%20smolagent/main.py)

#### 🔹 Day 14: RAG Smolagent with LangChain Retrievers
* **Focus**: Bridging retrieval-augmented documentation with Smolagent code generation.
* **Key Concept**: Created a custom `ProjectDetailsRetriverTool` class inheriting from `Tool`. Preprocessed custom project metadata using LangChain's `RecursiveCharacterTextSplitter`, indexed them in a BM25 retriever (`BM25Retriever`), and bound it to a `CodeAgent` to fetch and list specific project details.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2014%20Rag%20agent%20Smolagent/main.py)

#### 🔹 Day 15: LlamaIndex Pipelines & ChromaDB Integration
* **Focus**: Setting up robust vector database ingestion pipelines.
* **Key Concept**: Setup LlamaIndex `IngestionPipeline` incorporating a `SentenceSplitter` and `HuggingFaceEmbedding` (BGE small model). Ingested documents asynchronously and stored the generated vector embeddings in a persistent ChromaDB instance (`ChromaVectorStore`).
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2015%20LlamaIndex/main.py), [main-chroma.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2015%20LlamaIndex/main-chroma.py)

#### 🔹 Day 16: Custom Tools & ToolSpecs in LlamaIndex
* **Focus**: Exposing Python utilities to agentic interfaces.
* **Key Concept**: Learned how to package functions for agents. Implemented a `FunctionTool` by wrapping a plain Python function and utilized prebuilt specs (`GmailToolSpec`) to convert them to tool definitions.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2016%20LlamaIndex%20tools/main.py), [main-google.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2016%20LlamaIndex%20tools/main-google.py)

#### 🔹 Day 17: AgentWorkflows in LlamaIndex
* **Focus**: Designing conversational search assistants with Gemini and LlamaIndex.
* **Key Concept**: Wrapped LlamaIndex query engines inside a `QueryEngineTool` and unified it with external custom function tools. Built an `AgentWorkflow` that leverages Google's `gemini-3-flash-preview` to decide which tool to call to solve user queries.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2017%20LlamaIndex%20agents/main.py)

#### 🔹 Day 18: LlamaIndex Multi-Agent Systems
* **Focus**: Orchestrating multiple specialised sub-agents.
* **Key Concept**: Configured multiple independent `ReActAgent` instances: a `calculator_agent` (bound with math tools) and an `info_agent` (bound with personal project search tools). Managed them using a master `AgentWorkflow` routing agent, enabling seamless routing and collaboration between agents.
* **Files**: [multi-agent.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2018%20LLamaIndex%20multi%20agent/multi-agent.py)

#### 🔹 Day 19: Event-Driven LlamaIndex Workflows
* **Focus**: Coding robust event-driven workflows with branching and visual analysis.
* **Key Concept**: Built a custom `Workflow` using LlamaIndex workflows. Defined custom `Event` objects, decorated async methods with `@step` to receive particular events, integrated a random branch routing (`LoopEvent`), and drew visual pipeline graphs using `draw_all_possible_flows`.
* **Files**: [main-workflow.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2019%20LLama%20workflow/main-workflow.py), [main-loop.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2019%20LLama%20workflow/main-loop.py)

#### 🔹 Day 20: Event Summarizer Agentic Pipeline
* **Focus**: Solving complex real-world challenges using Multi-Agent coordination.
* **Key Concept**: Built a complete agentic system using `AgentWorkflow`. The `calendar_agent` uses `GoogleCalendarToolSpec` to search calendar meetings and upcoming deadlines. The `gmail_agent` takes the retrieved details, injects them into a premium HTML email template, and sends a scheduled summary using SMTP.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2020%20Event%20summarizer/main.py)

---

### 🕸️ Phase 4: Advanced LangGraph States & Persistence (Days 21–30)

#### 🔹 Day 21: Getting Started with LangGraph
* **Focus**: Implementing stateful cyclic graphs in LangGraph.
* **Key Concept**: Configured a `StateGraph` using a basic `TypedDict` state schema. Registered nodes modifying string states, added static edges, created a conditional router (`decide_mood`), compiled the graph, and saved the visualization as a Mermaid PNG.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2021%20LangGraph/main.py)

#### 🔹 Day 22: LangGraph Chains with Tool-Bound LLMs
* **Focus**: Designing single-step LLM tool calls.
* **Key Concept**: Configured `ChatGoogleGenerativeAI` with custom function tools using `llm.bind_tools([give_quote])`. Designed a simple `MessagesState` graph that invokes the tool-bound model and returns the message, leveraging standard reducers.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2022%20LangGraph%20Chain/main.py)

#### 🔹 Day 23: Prebuilt ToolNodes & Causal Routers
* **Focus**: Automating tool routing within state graphs.
* **Key Concept**: Integrated LangGraph's prebuilt `ToolNode` and `tools_condition` router to automatically detect if the LLM's response contains tool calls and route execution accordingly.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2023%20LangGraph%20Routes/main.py)

#### 🔹 Day 24: Cyclic ReAct Agent Loops
* **Focus**: Enabling agents to loop and resolve multiple tools sequentially.
* **Key Concept**: Built a dynamic mathematical agent graph. Binds multiple math tools (`multiply`, `add`) to the model, and creates a loop (`START -> call_llm -> tools_condition -> ToolNode -> call_llm`), allowing the agent to solve compound math queries step-by-step.
* **Files**: [main.py](file:///f:/AI%20ML/AI-ML-Complete/Day%2024%20LangGraph%20simple%20Agent/main.py)

#### 🔹 Day 25: LangGraph Short-Term Memory
* **Focus**: Maintaining state and context across multiple user inputs.
* **Key Concept**: Integrated checkpointers to enable short-term thread memory. Compiled the StateGraph with `MemorySaver()`, allowing the agent to remember context (e.g. intermediate results) when running consecutive invocations under the same `thread_id`.
* **Files**: [main.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2025%20LangGraph%20Memory/main.ipynb)

#### 🔹 Day 26: Strict State Validation via Pydantic
* **Focus**: Enforcing schema boundaries at graph runtimes.
* **Key Concept**: Replaced basic `TypedDict` states with a schema inheriting from `pydantic.BaseModel` (`PydanticState`). Configured a `@field_validator` to enforce strict literal checks, ensuring that invalid inputs throw standard `ValidationError` runtime errors instead of bypassing typings silently.
* **Files**: [main.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2026%20State%20Schema/main.ipynb)

#### 🔹 Day 27: State Reducers & Parallel Merges
* **Focus**: Customizing how states accumulate and merge updates.
* **Key Concept**: Analyzed state updates in parallel branching models. Implemented custom reducer functions (`custome_reducer`) to safely accumulate integer lists, handling empty states and merging concurrent parallel outputs cleanly.
* **Files**: [main.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2027%20LangGraph%20state%20Reducers/main.ipynb)

#### 🔹 Day 28: Input, Output, and Overall State Separation
* **Focus**: Encapsulating agent variables and exposing clean APIs.
* **Key Concept**: Created separate state schemas for the agent graph: an `InputState` containing only initial variables, an `OverallState` containing scratchpad variables, and an `OutputState` containing the final output. Compiled the StateGraph with `input_schema` and `output_schema` to hide scratchpad variables from end users.
* **Files**: [main.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2028%20Mutilpe%20States/main.ipynb)

#### 🔹 Day 29: Chatbot Summarizer (Context Window Management)
* **Focus**: Preserving chat context over extremely long conversation threads.
* **Key Concept**: Programmed a conversation management chatbot in LangGraph. Built a conditional router that triggers a `summarize_messages` step when message history exceeds a threshold. The summarizer compresses previous conversations into a `summary` property and drops older messages using `RemoveMessage` to maintain minimal token counts.
* **Files**: [main.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2029%20Chatbot%20summizer/main.ipynb)

#### 🔹 Day 30: Persistent SQLite Checkpointers
* **Focus**: Building production-grade permanent conversational memory.
* **Key Concept**: Replaced the in-memory memory checkpointer with a database-backed checkpointer. Integrated sqlite3 `connection` with LangGraph's `SqliteSaver`, storing checkpoints permanently in `chatbot.db`. Conversation summaries and threads are fully retained and loaded across system restarts.
* **Files**: [main.ipynb](file:///f:/AI%20ML/AI-ML-Complete/Day%2030%20chatbot%20external%20memory/main.ipynb)

---

## 🛠️ Key Technologies Used

* **Deep Learning**: PyTorch (`nn.Module`, Tensors), TensorFlow / Keras, NumPy (Matrix calculations from scratch)
* **Agentic Frameworks**: LangGraph (StateGraph, Checkpointers, Reducers), Smolagents (HF), LlamaIndex (AgentWorkflow, Workflows)
* **Vector Store & Retrievers**: ChromaDB, BM25 (LangChain), HuggingFace Embeddings
* **Tokenizers & Tools**: Hugging Face Tokenizers (BPE), SQLite (Persistence), Pydantic (Validation), Google API (Calendar)