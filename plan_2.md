# 🧠 40-Day GenAI & LLM Engineering Advanced Plan


## Topic 1: Transformers Architecture

**Task**: Master the core architectural building blocks of the original Transformer model, focusing on attention mechanisms, positional encodings, layer normalization, and the mathematical mechanics of sequence-to-sequence learning.

**List of things to study**
- 🔹 Self-Attention mechanism mathematical formulation ($Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$).
- 🔹 Multi-Head Attention vs. Single-Head Attention.
- 🔹 Positional Encodings (sinusoidal functions vs. learned embeddings).
- 🔹 Encoder and Decoder stack differences (masked self-attention in decoder).
- 🔹 Normalization layers (Pre-LN vs. Post-LN) and residual connections.
- 📚 Resource: "Attention Is All You Need" paper by Vaswani et al.

**Relevant Questions to Answer**
1. What is the query, key, and value concept in self-attention, and how do they map to database-style retrievals?
2. Why is the dot-product in self-attention scaled by the square root of the key dimension ($d_k$)? What happens if this scaling factor is omitted?
3. How does masked self-attention prevent the decoder from looking at future tokens during training?
4. What is the difference between Pre-LN and Post-LN architectures, and why do modern LLMs almost exclusively use Pre-LN?
5. How do sinusoidal positional encodings represent relative sequence positions without training parameters?
6. What is the computational complexity of standard self-attention relative to input sequence length, and why does this make long-context training difficult?
7. What is the role of the Feed-Forward Network (FFN) layer within each Transformer block, and how does its parameter count compare to the attention layer?
8. Why do we need multi-head attention instead of just computing one large single-head attention?

---

## Topic 2: Modern LLM Architecture (MLA, MoE, KV Cache)

**Task**: Understand the structural evolution of modern LLMs, focusing on how Multi-head Latent Attention and Mixture of Experts optimize compute while managing the KV Cache bottleneck during inference.

**List of things to study**
- 🔹 Token generation (Decode) vs Prompt processing (Prefill).
- 🔹 Mixture of Experts (MoE) routing mechanisms.
- 🔹 KV Cache memory consumption formulas.
- 🛠️ Sketch latency breakdown for Multi-head Latent Attention (MLA).
- 📚 Resource: DeepSeek-V3 or Llama 3 architecture papers.

**Relevant Questions to Answer**
1. What is the "prefill" phase in LLM inference, and why is it compute-intensive?
2. Why is the "decode" phase memory-bandwidth bound?
3. How much memory does the KV cache consume for a 7B model with 2048 context length?
4. How does Mixture of Experts (MoE) reduce active FLOPs during inference while increasing total VRAM requirements?
5. Why do longer generations take linearly more time, but longer prompts take super-linear time?
6. How does Multi-head Latent Attention (MLA) compress the KV Cache compared to standard Multi-Head Attention (MHA)?
7. What is the "memory wall" in the context of LLMs?
8. What is the theoretical max tokens/sec for a given GPU memory bandwidth?

---

## Topic 3: Local Reasoning Models (DeepSeek-R1 & Ollama)

**Task**: Deploy and interact with local reasoning models to observe chain-of-thought processing, analyzing resource consumption, and testing execution speed on local hardware setups.

**List of things to study**
- 🔹 Ollama architecture and backend (llama.cpp).
- 🔹 Reasoning models vs. Instruct models.
- 🔹 VRAM vs System RAM allocation.
- 🛠️ Run deepseek-r1:7b via Ollama and monitor the <thought> tag processing.
- 📚 Resource: Ollama documentation and llama.cpp GitHub repository.

**Relevant Questions to Answer**
1. What backend does Ollama use to run models locally?
2. How does Ollama manage model downloads, storage, and caching?
3. What happens if your system runs out of VRAM during local inference?
4. How does a reasoning model (like DeepSeek-R1) utilize <thought> tags before generating a final response?
5. Why does the first response in a local model often take longer than subsequent ones?
6. How would you measure tokens/sec manually using timestamps?
7. What system metrics (CPU, RAM, swap) should you monitor during local inference?
8. How does Ollama handle dynamic offloading of layers between GPU and CPU?

---

## Topic 4: Advanced Quantization (GGUF, EXL2, AWQ)

**Task**: Master quantization techniques that reduce model footprint and memory bandwidth requirements without severely degrading reasoning capabilities, allowing massive models to run on consumer hardware.

**List of things to study**
- 🔹 Numerical precision reduction (FP16 → INT8 → INT4).
- 🔹 GGUF vs GPTQ vs AWQ vs EXL2 tradeoffs.
- 🔹 Symmetric vs Asymmetric quantization.
- 🛠️ Compare file sizes and generation speeds of FP16 vs 4-bit GGUF models.
- 📚 Resource: HuggingFace Quantization guides.

**Relevant Questions to Answer**
1. What does “4-bit quantization” actually mean in terms of memory per parameter?
2. Why doesn’t quantization destroy model performance completely?
3. What is the role of calibration data in GPTQ and AWQ?
4. What’s the difference between symmetric and asymmetric quantization?
5. Why is GGUF popular for CPU/mixed inference, while EXL2 is preferred for high-speed GPU inference?
6. How much speedup can you expect from 4-bit vs FP16 on a memory-bound GPU?
7. What are common artifacts or reasoning failures in heavily quantized models?
8. Does quantization affect prompt processing (prefill) or token generation (decode) more?
9. Is it possible to de-quantize a model back to FP16 and regain its original accuracy?

---

## Topic 5: High-Throughput Serving with vLLM

**Task**: Implement production-grade LLM serving using vLLM to maximize GPU utilization through continuous batching and efficient memory management via PagedAttention.

**List of things to study**
- 🔹 Continuous vs Static batching.
- 🔹 PagedAttention architecture and virtual memory paging.
- 🔹 KV cache fragmentation.
- 🛠️ Deploy a vLLM Docker container and test its OpenAI-compatible API.
- 📚 Resource: vLLM Documentation and PagedAttention research paper.

**Relevant Questions to Answer**
1. What is the main bottleneck in naive Hugging Face LLM serving?
2. How does continuous batching improve GPU utilization compared to static batching?
3. What is “KV cache fragmentation,” and how does PagedAttention fix it?
4. How is PagedAttention conceptually similar to OS virtual memory paging?
5. What happens if two requests have very different prompt lengths in vLLM?
6. How does vLLM handle long-context prompts (e.g., 32K tokens) in terms of block allocation?
7. How do you specify --tensor-parallel-size in vLLM, and when is it necessary?
8. What is “swap space” in vLLM, and when is the system forced to use it?

---

## Topic 6: Speculative Decoding & Medusa

**Task**: Accelerate the memory-bound decode phase by using small draft models or multiple decoding heads to predict and verify multiple tokens simultaneously.

**List of things to study**
- 🔹 Draft model vs Target model architectures.
- 🔹 Token verification algorithms and acceptance rates.
- 🔹 Medusa architecture (multi-head decoding).
- 🛠️ Implement a local speculative decoding pipeline and measure the token/sec speedup.
- 📚 Resource: Medusa GitHub repository and Speculative Decoding papers.

**Relevant Questions to Answer**
1. How does speculative decoding bypass the memory-bandwidth bottleneck of the decode phase?
2. What are the requirements for a "draft model" in standard speculative decoding?
3. How does the target model verify the tokens proposed by the draft model in parallel?
4. What is the "acceptance rate" and why is it the critical metric for speculative decoding success?
5. How does the Medusa architecture differ from using a separate draft model?
6. Why is speculative decoding often less effective at very large batch sizes?
7. Can speculative decoding change the final output of a greedy decoding generation?
8. What is the VRAM overhead of implementing a speculative decoding pipeline?

---

## Topic 7: Embedding Models & Late Interaction (ColBERT)

**Task**: Differentiate between traditional dense embeddings and late interaction models like ColBERT to drastically improve retrieval precision for complex, nuanced queries.

**List of things to study**
- 🔹 Bi-encoders (Cosine Similarity) vs Cross-encoders.
- 🔹 Late Interaction architecture and ColBERT token-level matching.
- 🔹 Dimensionality and storage costs.
- 🛠️ Compare search results of all-MiniLM-L6-v2 vs ColBERTv2 on a complex lexical query.
- 📚 Resource: MTEB Leaderboard and ColBERT research paper.

**Relevant Questions to Answer**
1. Why do traditional Bi-encoders struggle with highly specific or lexical queries?
2. What is "Late Interaction" and how does ColBERT process queries differently than a standard sentence transformer?
3. What is the storage cost tradeoff when moving from dense embeddings to ColBERT?
4. How does cosine similarity actually work mathematically in vector search?
5. How do you handle out-of-vocabulary terms in embedding models?
6. Why is the choice of embedding model critical to the foundation of a RAG pipeline?
7. Can you mix embeddings from two different models in the same vector database?
8. How does ColBERT balance the speed of a Bi-encoder with the accuracy of a Cross-encoder?

---

## Topic 8: Hybrid Search & Reranking

**Task**: Combine keyword-based BM25 search with semantic vector search, applying a cross-encoder reranking step to solve the "lost in the middle" retrieval problem.

**List of things to study**
- 🔹 BM25 algorithms (Sparse search).
- 🔹 Reciprocal Rank Fusion (RRF).
- 🔹 Cross-encoder reranking and latency overhead.
- 🛠️ Implement an RRF pipeline combining keyword search with dense vectors, followed by a Cohere/BGE reranker.
- 📚 Resource: Pinecone Hybrid Search Guide.

**Relevant Questions to Answer**
1. What specific retrieval failures does BM25 solve that vector search misses?
2. How does Reciprocal Rank Fusion (RRF) mathematically combine sparse and dense search results?
3. Why are cross-encoders more accurate for scoring (query, passage) pairs than bi-encoders?
4. What is the latency cost of running 100 documents through a cross-encoder reranker?
5. How many documents should you retrieve initially before passing them to the reranker?
6. Can you cache cross-encoder results for frequent queries?
7. How do you handle long documents that exceed the cross-encoder's token limit?
8. What is the difference between monoT5 and a standard cross-encoder for reranking?

---

## Topic 9: Semantic & Hierarchical Chunking

**Task**: Move beyond naive character splitting by implementing context-aware chunking strategies like Parent-Document Retrieval to maintain semantic integrity.

**List of things to study**
- 🔹 Naive chunking (RecursiveCharacterTextSplitter).
- 🔹 Parent-Document Retriever concepts (Small-to-Big retrieval).
- 🔹 Semantic boundary chunking.
- 🛠️ Implement a chunking script that preserves document metadata (like chapter titles) across chunks.
- 📚 Resource: LangChain Advanced Retrieval Docs.

**Relevant Questions to Answer**
1. What happens to retrieval accuracy if you set chunk_size=1000 with overlap=0?
2. How does a Parent-Document Retriever balance tight semantic matching with broad LLM context?
3. How do you avoid splitting a document in the middle of a sentence or a code block?
4. What is semantic chunking, and how does it use embeddings to find split boundaries?
5. How do you preserve essential metadata (like chapter titles) across chunks?
6. What is the tradeoff between small vs large chunks in a standard RAG pipeline?
7. How would you chunk a complex legal contract differently than a standard blog post?
8. How do you validate that no critical information was lost or orphaned during the chunking process?

---

## Topic 10: Vector Databases at Scale (Qdrant/Pinecone)

**Task**: Deploy and query a managed, highly scalable vector database, utilizing metadata payloads and advanced indexing algorithms like HNSW for fast retrieval.

**List of things to study**
- 🔹 Vector indexing algorithms (HNSW vs IVF).
- 🔹 Payload/Metadata filtering.
- 🔹 In-memory vs Disk-backed storage architectures.
- 🛠️ Spin up a Qdrant cluster and insert 10,000 vectors with distinct metadata payloads for pre-filtering.
- 📚 Resource: Qdrant Documentation and HNSW algorithms.

**Relevant Questions to Answer**
1. How does HNSW (Hierarchical Navigable Small World) achieve fast approximate nearest neighbor search?
2. What is the difference between "indexing time" and "query time" in vector databases?
3. How do you apply metadata filtering (pre-filtering vs post-filtering) without ruining recall?
4. What happens to query latency as a vector database scales past 10 million vectors?
5. Why is a dedicated vector database like Qdrant preferred over an in-memory FAISS index in production?
6. How do you normalize embeddings before storing them in the database?
7. Can you update or delete specific vectors efficiently in HNSW?
8. How would you handle a multi-tenant vector database architecture?

---

## Topic 11: GraphRAG & Knowledge Graphs

**Task**: Map complex relationships within data using GraphRAG, extracting entities and communities to answer global, multi-hop queries that defeat standard vector search.

**List of things to study**
- 🔹 Entity and Relationship extraction via LLMs.
- 🔹 Knowledge graph structures (Nodes/Edges).
- 🔹 Cypher query language (Neo4j).
- 🛠️ Use an LLM to convert unstructured text into a set of nodes and edges, and store them in Neo4j.
- 📚 Resource: Microsoft GraphRAG repository.

**Relevant Questions to Answer**
1. What type of queries completely fail in standard RAG but succeed in GraphRAG?
2. How do you use an LLM to automatically extract nodes and edges from unstructured text?
3. What is the concept of "Community Summarization" in Microsoft's GraphRAG approach?
4. How do you write a prompt to convert a user's natural language query into a valid Cypher query?
5. What is the latency and cost overhead of building a knowledge graph compared to vector embeddings?
6. How does GraphRAG handle contradictory relationships extracted from different documents?
7. Can you combine vector search with graph traversals in a single query?
8. How do you evaluate the accuracy of an LLM-generated knowledge graph?

---

## Topic 12: Long-Context "RAG" (The 1M Token Era)

**Task**: Evaluate the cost, latency, and accuracy trade-offs between utilizing massive 1M+ context windows versus traditional RAG pipelines for document Q&A.

**List of things to study**
- 🔹 Context window scaling limits and KV Cache explosion.
- 🔹 "Lost in the Middle" phenomenon.
- 🔹 Cache-augmented generation (Prompt Caching).
- 🛠️ Run a needle-in-a-haystack test using Gemini 1.5 Pro or Claude 3.5 with a 500-page PDF.
- 📚 Resource: Anthropic Prompt Caching Guide.

**Relevant Questions to Answer**
1. Why not just put a 500-page PDF into a 1M token context window instead of building RAG?
2. What is the "Lost in the Middle" problem, and do modern long-context models still suffer from it?
3. How does Prompt Caching (e.g., in Anthropic's API) change the economics of long-context LLMs?
4. What is the latency impact of sending 500k tokens in a single prefill phase?
5. How do you evaluate whether a model actually understands the 1M tokens or is just skimming?
6. Can long-context models replace cross-encoder rerankers?
7. In what specific scenarios does RAG outperform a 1M token context window?
8. How does Needle-In-A-Haystack (NIAH) testing work for long-context models?

---

## Topic 13: RAG Evaluation (RAGAS & DeepEval)

**Task**: Systematically measure retrieval and generation quality using automated frameworks to calculate faithfulness, answer relevance, and context precision.

**List of things to study**
- 🔹 Component-wise evaluation (Retrieval vs Generation).
- 🔹 Faithfulness and Answer Relevance metrics.
- 🔹 Context Precision and Context Recall.
- 🛠️ Generate a synthetic ground-truth dataset and run RAGAS against your local RAG pipeline.
- 📚 Resource: RAGAS Documentation.

**Relevant Questions to Answer**
1. How do you evaluate retrieval quality completely separately from generation quality?
2. What does the "Faithfulness" metric measure in the RAGAS framework?
3. How does "Context Recall" detect if your retriever missed critical information?
4. What is the "LLM-as-a-Judge" bias, and how do you mitigate it?
5. Why are traditional metrics like BLEU or ROUGE insufficient for evaluating RAG?
6. How do you generate a synthetic ground-truth dataset for evaluating RAG?
7. What happens to evaluation scores if the retrieved context contradicts the LLM's internal knowledge?
8. How many test queries do you need to confidently evaluate a RAG pipeline's production readiness?

---

## Topic 14: Building the RAG API (FastAPI)

**Task**: Expose the complete retrieval and generation pipeline as a robust, asynchronous REST API with streaming support (Server-Sent Events) and error handling.

**List of things to study**
- 🔹 FastAPI routing and Pydantic models.
- 🔹 Asynchronous execution (async def).
- 🔹 Server-Sent Events (SSE) for token streaming.
- 🛠️ Write a /v1/chat/completions endpoint that streams chunks of text back to the client as they are generated.
- 📚 Resource: FastAPI documentation and Starlette SSE tools.

**Relevant Questions to Answer**
1. Why is asynchronous programming (async/await) critical when building a RAG API?
2. How do you implement Server-Sent Events (SSE) in FastAPI to stream tokens back to the user?
3. What HTTP status codes should you return for retrieval failures vs LLM timeouts?
4. How do you pass configuration (e.g., top_k, temperature) safely from the API client to the RAG engine?
5. How would you handle a scenario where the vector database connection drops during an API call?
6. What is the role of Pydantic in validating the incoming API request payload?
7. How do you trace a single user request end-to-end through the API, Retriever, and LLM?
8. How would you add basic rate limiting to this FastAPI endpoint?

---

## Topic 15: Tool Calling & Function Calling

**Task**: Enable LLMs to interact with external environments by defining strict JSON schemas and Pydantic models for reliable and structured tool execution.

**List of things to study**
- 🔹 Function calling APIs (OpenAI format).
- 🔹 Pydantic schema definition and JSON Schema extraction.
- 🔹 Tool descriptions and docstrings.
- 🛠️ Create a Python function to check weather, bind it to an LLM, and parse the resulting tool call.
- 📚 Resource: OpenAI Function Calling Guide.

**Relevant Questions to Answer**
1. How does the LLM mathematically know when to output a tool call versus standard text?
2. Why are the docstrings/descriptions of your tools just as important as the code inside them?
3. What happens if an API tool fails or times out? How should the LLM be notified?
4. Can an agent call multiple tools in parallel in a single turn?
5. How do you pass complex, nested JSON objects as arguments to a tool?
6. How do you force an LLM to use a specific tool (tool choice)?
7. How do you handle type mismatches (e.g., LLM outputs a string instead of an int)?
8. What is the security risk of giving an LLM a tool that executes raw SQL?

---

## Topic 16: Stateful Agents with LangGraph

**Task**: Design cyclic, graph-based agent workflows that maintain state across complex, multi-step tasks, replacing fragile linear chains with robust state machines.

**List of things to study**
- 🔹 Nodes, Edges, and Conditional Edges.
- 🔹 State Management (StateGraph and TypedDict).
- 🔹 Cyclic execution workflows and infinite loop prevention.
- 🛠️ Build a simple Research Agent in LangGraph that loops between a "Search" node and an "Evaluate" node until the answer is satisfactory.
- 📚 Resource: LangGraph Documentation.

**Relevant Questions to Answer**
1. Why is a graph-based architecture (LangGraph) superior to a linear chain (LangChain) for agents?
2. How is "State" passed and updated between nodes in LangGraph?
3. What is a "Conditional Edge," and how does it enable routing logic?
4. How do you prevent an agent from getting stuck in an infinite loop?
5. What is the latency overhead of running a state machine versus a raw prompt loop?
6. How do you checkpoint and persist state so a user can resume a task later?
7. Can LangGraph execute multiple nodes in parallel?
8. How do you debug state transitions when an agent makes the wrong decision?

---

## Topic 17: Multi-Agent Orchestration (CrewAI / AutoGen)

**Task**: Architect systems where multiple specialized agents collaborate, delegate tasks, and debate to solve complex problems faster than a single monolithic agent.

**List of things to study**
- 🔹 Agent personas, roles, and backstories.
- 🔹 Task delegation and sequential vs hierarchical processes.
- 🔹 Inter-agent communication protocols.
- 🛠️ Define a "Writer" agent and an "Editor" agent in CrewAI and have them collaborate on a blog post.
- 📚 Resource: CrewAI Documentation.

**Relevant Questions to Answer**
1. What is the difference between a multi-agent system and a single agent with multiple tools?
2. How do CrewAI and AutoGen differ in their approach to agent conversation routing?
3. What is the token cost implication of having multiple agents review each other's work?
4. How do you prevent agents from hallucinating agreements in a debate setting?
5. Can you mix local open-source models and API models within the same multi-agent team?
6. How do you handle context window limits as agent conversations grow long?
7. What is a hierarchical process, and how does a "Manager Agent" allocate tasks?
8. How do you evaluate the overall output quality of a multi-agent system?

---

## Topic 18: Reasoning Loops (Self-Correction & Reflection)

**Task**: Implement meta-prompts and reflection patterns that force the model to critique its own initial output and iteratively self-correct before presenting a final answer.

**List of things to study**
- 🔹 Chain-of-Thought (CoT) prompting techniques.
- 🔹 The "Actor-Critic" LLM pattern.
- 🔹 Self-reflection prompts.
- 🛠️ Write a prompt chain where output from Step 1 is fed into Step 2 with the prompt: "Find three flaws in the previous reasoning and rewrite it."
- 📚 Resource: "Reflexion: Language Agents with Verbal Reinforcement" paper.

**Relevant Questions to Answer**
1. Why does an LLM often fail to generate a perfect answer on the first try but succeed when asked to critique itself?
2. What is the difference between Chain-of-Thought and Self-Reflection?
3. How do you format a reflection prompt to prevent the model from just blindly agreeing with its first output?
4. What is the latency impact of enforcing a self-correction loop in a production API?
5. How many reflection iterations should you allow before forcing a final output?
6. Can you use a smaller, faster model as the "Actor" and a larger, smarter model as the "Critic"?
7. How do you evaluate whether the self-correction loop actually improved the metric (e.g., accuracy)?
8. What happens if the critic model introduces new hallucinations during the correction phase?

---

## Topic 19: Agent Memory (Short-term vs. Long-term)

**Task**: Equip agents with persistence by distinguishing between conversation buffer memory (short-term) and vector/SQL-backed semantic memory (long-term).

**List of things to study**
- 🔹 Conversation Buffer Memory vs Summary Memory.
- 🔹 Entity extraction for long-term user profiles.
- 🔹 Semantic search across past interactions.
- 🛠️ Implement SQLite checkpointing in LangGraph to allow a conversation to pause and resume the next day.
- 📚 Resource: Mem0 or Zep documentation for agent memory.

**Relevant Questions to Answer**
1. How does conversation history get injected into an agent’s prompt without overflowing the context window?
2. What is the difference between episodic memory and semantic memory in the context of LLMs?
3. How do you use a vector database to give an agent "long-term memory" of past sessions?
4. How do you prevent memory injection from causing prompt drift or confusing the agent's instructions?
5. What strategies exist for updating or deleting outdated facts in an agent's long-term memory?
6. How do you handle multi-tenant memory (keeping User A's memory completely isolated from User B)?
7. Can an agent proactively decide to "write" something to its long-term memory via a tool call?
8. How does long-term memory impact the latency of the agent's prefill phase?

---

## Topic 20: Planning & Task Decomposition

**Task**: Prevent agent paralysis on complex requests by implementing Plan-and-Execute architectures that break massive goals into manageable, sequential sub-tasks.

**List of things to study**
- 🔹 Plan-and-Execute vs ReAct methodologies.
- 🔹 LLMCompiler routing.
- 🔹 Sub-task dependency mapping (DAGs).
- 🛠️ Write a prompt that takes a user goal (e.g., "Research Apple and write a financial report") and outputs a strict 5-step JSON plan.
- 📚 Resource: "Plan-and-Solve Prompting" paper.

**Relevant Questions to Answer**
1. Why do standard ReAct agents struggle with tasks that require more than 4 or 5 steps?
2. How does a Plan-and-Execute architecture separate the planning phase from the action phase?
3. What happens if a step in the initial plan fails? How does the agent replan?
4. How do you pass the context/output of Sub-task 1 into Sub-task 2?
5. Can you execute sub-tasks in parallel if they have no dependencies?
6. How do you constrain the planner model to only use available tools in its plan?
7. What is the role of an "Objective function" in agent task planning?
8. How do you display the agent's plan to the end-user in a UI for transparency?

---

## Topic 21: Evaluating Agents

**Task**: Measure non-deterministic agent trajectories to evaluate whether the agent chose the most efficient path, used the correct tools, and reached the right conclusion.

**List of things to study**
- 🔹 Agent Trajectory Evaluation.
- 🔹 Success Rate vs Efficiency metrics.
- 🔹 Tool selection accuracy.
- 🛠️ Use LangSmith to review a trace of a 5-step agent interaction and identify where it hallucinated a tool argument.
- 📚 Resource: LangChain Evaluation docs.

**Relevant Questions to Answer**
1. Why is evaluating an agent exponentially harder than evaluating a simple classification model?
2. What is "Trajectory Evaluation," and what specifically does it measure?
3. How do you automatically score whether an agent used the correct tool for a given user prompt?
4. What happens if an agent achieves the correct final answer but took 10 unnecessary steps to get there?
5. How do you build a synthetic dataset to test an agent's edge cases and tool failure handling?
6. Can you use an LLM-as-a-judge to evaluate another agent's tool arguments?
7. What metrics should you track in production to monitor agent degradation?
8. How do you handle non-deterministic reasoning paths when writing unit tests for agents?

---

## Topic 22: Human-in-the-loop (HITL) Architectures

**Task**: Enforce security and safety by building interrupt points into agent workflows, requiring explicit human approval before executing sensitive actions (like sending an email or deleting data).

**List of things to study**
- 🔹 Interrupts and Checkpoints in LangGraph.
- 🔹 State approval and modification routing.
- 🔹 Tool sandboxing and isolation.
- 🛠️ Add an "Interrupt" node in your agent graph that halts execution until a user types "Y" to approve an API call.
- 📚 Resource: LangGraph HITL guide.

**Relevant Questions to Answer**
1. What specific types of agent actions mandate a Human-in-the-Loop architecture?
2. How do you technically pause an agent's execution state and resume it hours later after human approval?
3. Can a human reviewer modify the agent's generated action/payload before approving it?
4. What happens if the human rejects the action? How does the agent handle the rejection feedback?
5. How do you design a UI that clearly shows the human exactly what the agent is trying to do?
6. What is "Tool Sandboxing," and why is it a necessary complement to HITL?
7. How does HITL impact the end-to-end latency and user experience of an application?
8. Can you implement conditional HITL (e.g., only ask for approval if the transaction is over $100)?

---

## Topic 23: Fine-Tuning vs. RAG (The Decision Matrix)

**Task**: Master the strategic decision-making process to correctly identify when to invest in fine-tuning (for style/format) versus RAG (for knowledge/facts) to save time and budget.

**List of things to study**
- 🔹 Cost analysis (Compute vs Vector DB overhead).
- 🔹 Style transfer vs Knowledge injection.
- 🔹 Latency constraints for edge deployment.
- 🛠️ Create a 2x2 decision matrix mapping "Need for new facts" against "Need for specific formatting/tone."
- 📚 Resource: OpenAI's Fine-Tuning vs RAG technical blog.

**Relevant Questions to Answer**
1. Why is it generally a bad idea to fine-tune an LLM purely to teach it new facts?
2. In what scenarios is Prompt Engineering completely insufficient, mandating Fine-Tuning?
3. How do the recurring costs of a RAG pipeline (Vector DB, embeddings) compare to the upfront cost of fine-tuning?
4. Can you combine Fine-Tuning and RAG in the same system? What is the benefit?
5. If a model needs to output a highly specific proprietary JSON schema 100% of the time, should you use RAG or Fine-Tuning?
6. How does fine-tuning affect the context window limitations compared to RAG?
7. What is the impact on latency when comparing a fine-tuned local model vs a massive RAG retrieval step?
8. How quickly does the information in your system update, and how does that influence the RAG vs FT decision?

---

## Topic 24: Fast Fine-Tuning with Unsloth

**Task**: Radically accelerate the local training process and reduce VRAM requirements by utilizing the Unsloth library to fine-tune a Llama 3 or Mistral model efficiently.

**List of things to study**
- 🔹 Unsloth optimizations (Triton kernels, manual autograd).
- 🔹 Memory-efficient backpropagation.
- 🔹 Exporting to GGUF format post-training.
- 🛠️ Write an Unsloth script to fine-tune Llama 3.1 8B on a Google Colab free tier GPU.
- 📚 Resource: Unsloth GitHub repository and documentation.

**Relevant Questions to Answer**
1. How does Unsloth achieve 2x faster training speeds compared to standard Hugging Face Transformers?
2. What specific memory optimizations allow Unsloth to fine-tune an 8B model on a single 16GB GPU?
3. Why does Unsloth rewrite standard attention and LoRA matrices using custom Triton kernels?
4. Can you use Unsloth for full-parameter fine-tuning, or is it strictly for PEFT/LoRA?
5. How do you format your dataset (e.g., Alpaca or ChatML format) to be compatible with Unsloth?
6. What is the process for saving the fine-tuned adapter and merging it with the base model in Unsloth?
7. How does Unsloth handle 4-bit quantization during the training process?
8. Are there any quality tradeoffs when using Unsloth's optimized kernels compared to standard PyTorch?

---

## Topic 25: QLoRA & PEFT Deep Dive

**Task**: Understand the underlying mathematics and mechanics of Quantized Low-Rank Adaptation (QLoRA) and how it enables fine-tuning massive models by freezing base weights and training small rank matrices.

**List of things to study**
- 🔹 Low-Rank Adaptation (LoRA) mathematics ($W = W_0 + BA$).
- 🔹 4-bit NormalFloat (NF4) quantization.
- 🔹 Double Quantization and Paged Optimizers.
- 🛠️ Sketch the matrix multiplication flow showing frozen pretrained weights bypassing gradients while LoRA adapters update.
- 📚 Resource: QLoRA research paper by Tim Dettmers.

**Relevant Questions to Answer**
1. Why is the low-rank assumption valid for fine-tuning large language models?
2. What do the parameters r (rank) and lora_alpha actually control in a LoRA configuration?
3. How does QLoRA drastically reduce memory requirements compared to standard LoRA?
4. What is the role of 4-bit NormalFloat (NF4) precision in the QLoRA process?
5. Why are the LoRA adapters kept in FP16 or BF16 precision while the base model is quantized to 4-bit?
6. What happens if you apply LoRA adapters to all linear layers (MLP, Projections) instead of just the attention heads?
7. What is a "Paged Optimizer," and how does it prevent Out-Of-Memory (OOM) spikes during training?
8. Can you achieve the exact same accuracy with QLoRA as you can with full-parameter fine-tuning?

---

## Topic 26: Dataset Curation & Synthetic Data

**Task**: Construct high-quality, formatted training datasets, using larger "teacher" models (like GPT-4) to generate synthetic instruction-following data to train smaller local models.

**List of things to study**
- 🔹 Instruction tuning formats (Alpaca, ShareGPT, ChatML).
- 🔹 Synthetic data generation (Teacher-Student distillation).
- 🔹 Dataset filtering, deduplication, and quality control.
- 🛠️ Write a prompt for GPT-4o to generate 100 domain-specific training examples in a strict JSON format.
- 📚 Resource: DistilABEL or Magpie documentation.

**Relevant Questions to Answer**
1. Why is dataset quality infinitely more important than dataset quantity in modern LLM fine-tuning?
2. What are the common formatting structures (e.g., Instruction, Input, Output) used in instruction tuning?
3. How do you prompt a frontier model (GPT-4) to generate diverse and challenging synthetic data, avoiding repetitive outputs?
4. What is the legal and ethical implication of using OpenAI's API to train a competing open-source model?
5. How do you identify and filter out "bad" or toxic data points from a large scraped dataset?
6. What is the risk of "mode collapse" or losing base model knowledge when fine-tuning on a narrow dataset?
7. How do you handle class imbalance if your synthetic dataset contains mostly short answers and few long explanations?
8. What is the role of system prompts in your dataset, and how do they impact the final tuned model?

---

## Topic 27: DPO (Direct Preference Optimization)

**Task**: Align fine-tuned models to human preferences and specific behaviors using DPO, avoiding the massive complexity of traditional RLHF reward modeling.

**List of things to study**
- 🔹 RLHF (Reinforcement Learning from Human Feedback) bottlenecks.
- 🔹 DPO loss function mathematics.
- 🔹 Chosen vs Rejected response pair formatting.
- 🛠️ Format a mini-dataset with prompt, chosen, and rejected columns, and run a basic DPO training loop.
- 📚 Resource: Direct Preference Optimization (DPO) paper.

**Relevant Questions to Answer**
1. Why did the industry largely shift from RLHF to DPO for model alignment?
2. How does DPO mathematically bypass the need to train a separate Reward Model?
3. What is the difference between Supervised Fine-Tuning (SFT) and Preference Alignment (DPO)?
4. How do you format a dataset for DPO, and what makes a good "rejected" response?
5. What is the beta parameter in the DPO loss function, and how does it control deviation from the base model?
6. Can you apply DPO directly to a base model, or must you do Supervised Fine-Tuning first?
7. How do you use synthetic data (e.g., an LLM as a judge) to generate chosen/rejected pairs for DPO?
8. What happens to the model's creativity or entropy if it is over-optimized using DPO?

---

## Topic 28: Domain-Specific Fine-Tuning

**Task**: Adapt base models to highly specialized vertical domains (e.g., Medical, Legal, or Text-to-SQL) where generalist models fail due to vocabulary or structural constraints.

**List of things to study**
- 🔹 Vocabulary expansion vs Adapter training.
- 🔹 Text-to-SQL specific dataset structures.
- 🔹 Domain-specific evaluation metrics.
- 🛠️ Design a dataset schema specifically meant to teach a model to output raw SQL queries based on natural language questions.
- 📚 Resource: HuggingFace blogs on domain adaptation.

**Relevant Questions to Answer**
1. Why do general models like Llama 3 struggle with niche medical terminology or proprietary coding languages?
2. Should you continue pre-training on raw domain text, or jump straight to instruction tuning with domain Q&A pairs?
3. How do you prevent catastrophic forgetting of general knowledge (like basic grammar) when fine-tuning heavily on a specific domain?
4. In Text-to-SQL fine-tuning, how do you provide the database schema context within the training prompt?
5. What is the difference between teaching a model new domain facts vs teaching it a new domain format?
6. How do you evaluate a domain-specific model if public benchmarks (like MMLU) don't cover your specific niche?
7. Can you mix domain-specific data with general conversational data during training?
8. What size model is ideal for a highly constrained, single-task domain application?

---

## Topic 29: Merging LoRA Adapters

**Task**: Fuse fine-tuned LoRA weights back into the base model's physical weights to eliminate inference latency overhead and prepare the model for quantization or deployment.

**List of things to study**
- 🔹 Weight fusion math ($W_{merged} = W_0 + BA$).
- 🔹 Spherical Linear Interpolation (SLERP) and model merging techniques.
- 🔹 Multi-adapter routing.
- 🛠️ Write a script using the PEFT library to load a base model, attach a LoRA adapter, and execute merge_and_unload().
- 📚 Resource: MergeKit documentation.

**Relevant Questions to Answer**
1. What is the latency and memory overhead of running a base model with a separate LoRA adapter during inference?
2. How does the merge_and_unload() function mathematically alter the base model's weight matrices?
3. Once an adapter is merged, can the process be reversed to get the original base model back?
4. What happens if you try to merge multiple different LoRA adapters into the same base model?
5. What is task arithmetic, and how does it relate to merging multiple specialized models?
6. Why is merging adapters necessary before applying standard quantization (like GGUF) for local inference?
7. How do tools like MergeKit use techniques like SLERP or TIES to combine different models?
8. Can you dynamically load and swap different LoRA adapters at runtime without merging them?

---

## Topic 30: Evaluation Benchmarks (MMLU, HumanEval)

**Task**: Understand the landscape of public LLM benchmarks, how they are calculated, and how to identify data contamination when evaluating your own or third-party models.

**List of things to study**
- 🔹 MMLU (Massive Multitask Language Understanding).
- 🔹 HumanEval (Coding capabilities).
- 🔹 Data contamination and over-fitting in public leaderboards.
- 🛠️ Run the lm-evaluation-harness to calculate the exact MMLU score for a small local model.
- 📚 Resource: EleutherAI Language Model Evaluation Harness.

**Relevant Questions to Answer**
1. What exactly does the MMLU benchmark measure, and why is it the industry standard?
2. How is the HumanEval benchmark graded (is it multiple choice or execution-based)?
3. What is "data contamination," and how does it artificially inflate model benchmark scores?
4. Why are zero-shot benchmark scores often drastically lower than 5-shot or 8-shot scores?
5. How does the choice of prompt template affect a model's performance on a standardized benchmark?
6. Why should you not rely solely on public benchmarks to determine a model's fitness for your specific production use case?
7. What is the Elo rating system, and how does the LMSYS Chatbot Arena use it to rank models?
8. How do you create a private, uncontaminated benchmark for your company's proprietary use case?

---

## Topic 31: Small Language Models (SLMs) for Edge

**Task**: Optimize and deploy sub-3-Billion parameter models (like Phi-3 or Llama 3.2 1B) for extreme low-latency, offline, or mobile environments.

**List of things to study**
- 🔹 SLM architectures and tokenization efficiency.
- 🔹 Mobile deployment (CoreML, ONNX, ExecuTorch).
- 🔹 Over-training on dense, high-quality data (The "Textbooks are all you need" approach).
- 🛠️ Run Microsoft's Phi-3-Mini on a CPU, measure the RAM usage, and test its coding capabilities.
- 📚 Resource: "Textbooks Are All You Need" (Phi-1 research paper).

**Relevant Questions to Answer**
1. How did models like Phi-3 achieve performance comparable to much larger models despite having fewer than 4B parameters?
2. What is the role of synthetic, highly curated "textbook" data in training Small Language Models?
3. What are the absolute minimum RAM and compute requirements to run a 1B parameter model effectively?
4. How do SLMs perform on reasoning and logic tasks compared to simple summarization tasks?
5. Why are SLMs critical for mobile, IoT, or secure offline deployments?
6. How does the vocabulary size and tokenization strategy impact the efficiency of an SLM?
7. What are the limitations of SLMs regarding vast world knowledge compared to a 70B model?
8. How would you fine-tune a 1B model to act as a highly specialized router or classifier in a larger system?

---

## Topic 32: Observability & Tracing (LangSmith / Arize Phoenix)

**Task**: Implement deep tracing in your LLM application to capture prompt inputs, token usage, latency, tool calls, and exact execution paths for debugging non-deterministic failures.

**List of things to study**
- 🔹 Spans, traces, and LLM call hierarchy.
- 🔹 Token tracking and cost attribution.
- 🔹 Integration with LangChain/LangGraph.
- 🛠️ Instrument an agent script with LangSmith and inspect the visual trace of a failed tool call.
- 📚 Resource: LangSmith documentation and tracing concepts.

**Relevant Questions to Answer**
1. Why is standard software logging (e.g., Python logging) insufficient for debugging multi-step LLM agents?
2. What is the difference between a "trace" and a "span" in LLM observability?
3. How do you monitor the exact input prompt sent to the LLM after all RAG contexts and memory have been injected?
4. How can observability tools help you identify the specific step causing a latency bottleneck?
5. How do you track token usage and cost per user or per session?
6. Can you use observability data to automatically build a fine-tuning dataset (capturing good user interactions)?
7. How do you protect sensitive user data (PII) from being permanently stored in cloud tracing platforms?
8. What is the performance overhead of adding deep tracing to a production LLM API?

---

## Topic 33: Prompt Management & Versioning

**Task**: Treat prompts as mission-critical codebase components by implementing Git-backed version control, separating prompt logic from application code.

**List of things to study**
- 🔹 Prompt registries and templating (Jinja2).
- 🔹 Separation of prompt text from Python code.
- 🔹 A/B testing prompts.
- 🛠️ Refactor a Python script to load its system prompt dynamically from a versioned YAML file rather than hardcoding it as a string.
- 📚 Resource: Prompt Engineering frameworks (e.g., PromptLayer or Langfuse).

**Relevant Questions to Answer**
1. Why is hardcoding large prompt strings directly into your Python functions an anti-pattern?
2. How do you effectively version control prompts (e.g., using Git vs using a specialized prompt registry)?
3. What happens if a backend developer changes a prompt string, breaking the evaluation metrics? How do you roll back?
4. How do you manage different variations of a prompt for different models (e.g., a Claude prompt vs a Llama prompt)?
5. How do you implement A/B testing for two different prompt versions in production?
6. How do you document the expected input variables (like {context} or {user_query}) for a prompt template?
7. Can you use CI/CD pipelines to automatically run evaluation tests whenever a prompt file is updated?
8. How do you allow non-technical domain experts to edit prompts without giving them access to the source code?

---

## Topic 34: Cost & Latency Monitoring

**Task**: Build dashboards to track financial expenditures per token and measure Time to First Token (TTFT) to ensure unit economics and user experience remain viable.

**List of things to study**
- 🔹 Token economics (Input vs Output token pricing).
- 🔹 TTFT (Time to First Token) and TPOT (Time Per Output Token).
- 🔹 Streaming metrics.
- 🛠️ Build a lightweight Python decorator that calculates the estimated API cost and execution time for any LLM function call.
- 📚 Resource: OpenAI Pricing page and latency optimization guides.

**Relevant Questions to Answer**
1. Why are output tokens typically priced higher than input tokens by API providers?
2. What is TTFT (Time to First Token), and why is it the most critical metric for perceived user experience?
3. How do you calculate TPOT (Time Per Output Token), and what hardware constraints affect it?
4. How do you track and attribute costs when using local, self-hosted models (compute cost vs token cost)?
5. What is the impact of system prompts and RAG contexts on the per-request cost?
6. How do you set up alerts to prevent budget overruns caused by infinite agent loops?
7. How does enabling streaming affect your ability to measure latency metrics accurately?
8. What strategies can you implement to aggressively cache responses and lower overall API costs?

---

## Topic 35: Guardrails & Security (Llama Guard / NeMo)

**Task**: Protect the LLM application from prompt injections, jailbreaks, and toxic outputs by implementing input/output filtering models like Llama Guard.

**List of things to study**
- 🔹 Prompt injection and Jailbreak attack vectors.
- 🔹 Input/Output filtering architectures.
- 🔹 Llama Guard prompt formatting and taxonomy.
- 🛠️ Implement a middleware function that passes user queries through a Llama Guard model before sending them to the main LLM.
- 📚 Resource: Llama Guard research paper and NVIDIA NeMo Guardrails docs.

**Relevant Questions to Answer**
1. What is the difference between a prompt injection attack and a jailbreak?
2. How does Llama Guard use a taxonomy of safety categories to classify input prompts and output responses?
3. What is the latency cost of running a secondary guardrail LLM on every user request?
4. How do you handle false positives where the guardrail blocks a legitimate user request?
5. Why are traditional regex or keyword-based filters insufficient for securing LLMs?
6. How do NVIDIA NeMo Guardrails use semantic similarity to enforce topical boundaries (staying on topic)?
7. What is "Data Exfiltration" in the context of LLMs, and how can guardrails prevent it?
8. How do you securely deploy an LLM that has access to execute code or SQL queries?

---

## Topic 36: Dockerizing the LLM System

**Task**: Ensure consistent, reproducible deployments by packaging the LLM API, Vector Database, and frontend dashboard into a multi-container Docker Compose architecture.

**List of things to study**
- 🔹 Dockerfiles for Python APIs and GPU passthrough (NVIDIA Container Toolkit).
- 🔹 Docker Compose networking.
- 🔹 Volume mounts for model weight caching.
- 🛠️ Write a docker-compose.yml file that spins up FastAPI, Qdrant, and vLLM, ensuring they can communicate over a Docker network.
- 📚 Resource: NVIDIA Container Toolkit documentation.

**Relevant Questions to Answer**
1. Why is containerization critical when moving an LLM application from a laptop to a cloud server?
2. How do you enable a Docker container to access the host machine's GPUs?
3. What is the NVIDIA Container Toolkit, and how does the --gpus all flag work?
4. How do you prevent Docker from re-downloading massive 10GB model weights every time the container restarts?
5. How do you manage network communication between the API container and the Vector DB container using Docker Compose?
6. What base image should you use to minimize the size of your Python/FastAPI container?
7. How do you pass secure environment variables (like API keys) into the Docker containers?
8. What are the challenges of orchestrating GPU-bound containers in Kubernetes compared to Docker Compose?

---

## Topic 37: Semantic Caching (GPTCache)

**Task**: Drastically reduce latency and API costs by intercepting queries and returning cached LLM responses for semantically identical questions using vector similarity.

**List of things to study**
- 🔹 Exact match vs Semantic match caching.
- 🔹 Vector similarity thresholds for cache hits.
- 🔹 Cache invalidation and eviction policies.
- 🛠️ Integrate GPTCache into an API script so that asking "Who is the CEO of Apple?" and "Who currently runs Apple?" hit the same cache.
- 📚 Resource: GPTCache GitHub repository.

**Relevant Questions to Answer**
1. Why is standard dictionary caching (exact string match) largely ineffective for LLM chatbots?
2. How does Semantic Caching use embedding models to determine if two queries mean the same thing?
3. What happens if the similarity threshold for a cache hit is set too low?
4. What is the latency difference between a semantic cache hit and a full LLM generation cycle?
5. How do you handle cache invalidation for temporal questions (e.g., "What is the stock price of Apple today?")?
6. Can you cache intermediate steps of an agent's reasoning loop, or only the final response?
7. What backend storage systems are commonly used to scale semantic caches in production?
8. How does semantic caching impact the personalization of responses for different users?

---

## Topic 38: Structured Outputs (Instructor/Pydantic)

**Task**: Force unpredictable LLMs to strictly adhere to predefined JSON schemas, ensuring downstream application code never breaks due to parsing errors.

**List of things to study**
- 🔹 JSON Schema validation.
- 🔹 The Instructor library (patching LLM clients with Pydantic).
- 🔹 Grammar-constrained decoding (llama.cpp grammars).
- 🛠️ Define a complex Pydantic model (with nested lists) and use Instructor to force GPT-4o or a local model to populate it perfectly from unstructured text.
- 📚 Resource: Instructor library documentation.

**Relevant Questions to Answer**
1. Why is asking an LLM to "return only valid JSON" in the prompt notoriously unreliable?
2. How does the Instructor library use Pydantic models to guarantee the structure and types of the LLM output?
3. What happens at the API level (e.g., OpenAI's Structured Outputs feature) to force the model to comply with the schema?
4. What is "grammar-constrained decoding," and how does it prevent local models from generating invalid syntax at the token level?
5. How do you handle and retry errors when the LLM outputs data that fails Pydantic validation (e.g., a string instead of an integer)?
6. Can you use structured outputs to force an LLM to cite the exact source document for its claims?
7. How does enforcing a strict schema impact the latency and token usage of the generation?
8. Why is structured output absolutely critical when building autonomous agents that execute code?
