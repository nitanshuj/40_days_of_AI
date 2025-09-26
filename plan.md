# 🧠 40-Hour GenAI & LLM Engineering Foundations Plan

> Built for Nitanshu Joshi — Data Scientist & Aspiring LLM Systems Engineer  
> Goal: Master industrial-scale LLM engineering in 40 hours through hands-on projects, deep questions, and portfolio building.

---

## ⏰ Hour 1: LLM Inference — What’s the Bottleneck?

**Task**: Understand token generation vs prompt processing, KV Cache, and the memory-bandwidth bound nature of LLMs.

**List of things to study**  
- 🔹 Understand token generation vs prompt processing.  
- 🔹 Learn about KV Cache memory consumption.  
- 🔹 Why LLMs are memory-bandwidth bound, not compute-bound.  
- 🛠️ Sketch latency breakdown: prefill (prompt) vs decode (tokens).  
- 📚 Resource: [Latency in LLMs — Lilian Weng](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/#inference-efficiency)

**Relevant Questions to Answer**  
1. What is the "prefill" phase in LLM inference, and why is it compute-intensive?  
2. Why is the "decode" phase memory-bandwidth bound?  
3. How much memory does the KV cache consume for a 7B model with 2048 context length?  
4. What happens to latency when you double the prompt length?  
5. Why can’t we just use bigger GPUs to solve LLM inference bottlenecks?  
6. How does attention computation differ between prefill and decode?  
7. What is “memory wall” in the context of LLMs?  
8. Can model parallelism reduce KV cache memory pressure?  
9. Why do longer generations take linearly more time, but longer prompts take super-linear time?  
10. How does FlashAttention help during prefill but not decode?  
11. What is the role of HBM (High Bandwidth Memory) in LLM inference?  
12. How would you profile an LLM to identify if it’s memory- or compute-bound?  
13. Why do small models (e.g., Phi-2) feel “snappier” than large ones, even on CPU?  
14. What is the theoretical max tokens/sec for a given GPU memory bandwidth?  
15. How does quantization impact memory bandwidth requirements?

---

## ⏰ Hour 2: Install & Run Your First Local LLM

**Task**: Successfully install and interact with a local LLM using Ollama.

**List of things to study**  
- 🛠️ Install Ollama (`curl -fsSL https://ollama.com/install.sh | sh`).  
- 🛠️ Pull and run `mistral:7b-instruct` → `ollama run mistral:7b-instruct`.  
- 🛠️ Ask: “Explain quantum computing in simple terms.”  
- 🛠️ Note response time, tokens/sec (if shown), RAM usage via Activity Monitor/htop.  
- 💡 Pro Tip: Try `phi3:mini` for faster CPU inference.

**Relevant Questions to Answer**  
1. What backend does Ollama use to run models (e.g., llama.cpp)?  
2. Why does `mistral:7b-instruct` load faster than `llama3:8b`?  
3. How does Ollama manage model downloads and caching?  
4. Can you run multiple models simultaneously in Ollama?  
5. What happens if your system runs out of RAM during inference?  
6. How does Ollama handle GPU vs CPU execution?  
7. Where are Ollama models stored on disk?  
8. Can you specify context length in Ollama? How?  
9. Why does the first response take longer than subsequent ones?  
10. How would you measure tokens/sec manually using timestamps?  
11. What system metrics (CPU, RAM, swap) should you monitor during inference?  
12. Can you use Ollama as a library in Python?  
13. How does Ollama’s API mode work (`ollama serve`)?  
14. What license does Mistral-7B have? Can you use it commercially?  
15. How would you compare Ollama to LM Studio or GPT4All?

---

## ⏰ Hour 3: Quantization Demystified

**Task**: Understand how quantization reduces LLM size and inference cost while preserving performance.

**List of things to study**  
- 🔹 What is quantization? Reducing numerical precision (FP16 → INT8 → INT4).  
- 🔹 GGUF (llama.cpp) vs GPTQ (AutoGPTQ) vs AWQ — tradeoffs.  
- 🔹 Why we lose minimal accuracy but gain huge speed + memory savings.  
- 📚 Resource: [TheQuantumStatistician — LLM Quantization Explained (YouTube)](https://www.youtube.com/watch?v=OqTgRZxhH9k).  
- 🛠️ Compare file sizes: `Mistral-7B-v0.1-FP16 (~14GB)` vs `Mistral-7B-Instruct-v0.2-GGUF-Q4_K_M (~4.5GB)`.

**Relevant Questions to Answer**  
1. What does “4-bit quantization” actually mean in terms of memory per parameter?  
2. Why doesn’t quantization destroy model performance completely?  
3. What is the role of calibration data in GPTQ?  
4. Can you quantize a model after fine-tuning? Should you?  
5. What’s the difference between symmetric and asymmetric quantization?  
6. Why is GGUF popular for CPU inference, while GPTQ is preferred for GPU?  
7. What is “quantization-aware training” (QAT), and is it used in LLMs today?  
8. How much speedup can you expect from 4-bit vs FP16 on a CPU? On a GPU?  
9. What are common artifacts or failures in heavily quantized models?  
10. Can you run a 7B 4-bit model on a laptop with 16GB RAM? Why or why not?  
11. What is “k-quants” in GGUF (e.g., Q4_K_M)?  
12. Does quantization affect prompt processing or token generation more?  
13. Is it possible to de-quantize a model back to FP16?  
14. Why do some quantized models refuse to load on certain hardware?  
15. When would you choose AWQ over GPTQ for deployment?

---

## ⏰ Hour 4: vLLM Quickstart — High-Throughput Serving

**Task**: Learn how vLLM enables high-throughput LLM serving via PagedAttention and continuous batching.

**List of things to study**  
- 🔹 What is continuous batching? How does it differ from static batching?  
- 🔹 What problem does PagedAttention solve in KV cache management?  
- 🔹 Why is vLLM significantly faster than Hugging Face Transformers for serving?  
- 🛠️ Install vLLM: `pip install vllm`.  
- 🛠️ Start OpenAI-compatible server with a GGUF model (if supported) or FP16 model.  
- 🛠️ Test with `curl http://localhost:8000/v1/models`.

**Relevant Questions to Answer**  
1. What is the main bottleneck in naive LLM serving?  
2. How does continuous batching improve GPU utilization?  
3. What is “KV cache fragmentation,” and how does PagedAttention fix it?  
4. Can vLLM serve quantized models (GGUF/GPTQ)? Which formats are supported?  
5. Why does vLLM expose an OpenAI-compatible API?  
6. What happens if two requests have very different prompt lengths in vLLM?  
7. How does vLLM handle long-context prompts (e.g., 32K tokens)?  
8. What is the minimum GPU VRAM needed to run Mistral-7B with vLLM?  
9. Can vLLM run on CPU? Why or why not?  
10. How do you specify `--tensor-parallel-size` in vLLM, and when would you use it?  
11. What metrics does vLLM log by default?  
12. How does vLLM compare to TGI (Text Generation Inference)?  
13. Can you use vLLM with custom models (e.g., your fine-tuned Phi-2)?  
14. What is “swap space” in vLLM, and when is it used?  
15. How would you monitor vLLM in production (latency, throughput, errors)?

---

## ⏰ Hour 5: Build a FastAPI Wrapper for vLLM

**Task**: Create a production-style API layer that abstracts vLLM behind a clean, customizable endpoint.

**List of things to study**  
- 🛠️ Create `main.py` with FastAPI app.  
- 🛠️ Use `requests` to proxy to vLLM’s OpenAI-compatible server.  
- 🛠️ Add error handling (e.g., timeout, model not loaded).  
- 🛠️ Run with `uvicorn main:app --reload`.  
- 🛠️ Test with `curl` POST request.

**Relevant Questions to Answer**  
1. Why wrap vLLM instead of using its API directly?  
2. What are the risks of exposing vLLM’s raw API to clients?  
3. How would you add authentication to this endpoint?  
4. How do you handle long-running requests (e.g., 1000 tokens)?  
5. Can you add rate limiting in FastAPI? How?  
6. What HTTP status codes should you return for errors?  
7. How would you log request IDs for tracing across services?  
8. Should you validate the `prompt` input? What checks would you add?  
9. How do you pass `max_tokens`, `temperature`, etc., from client to vLLM?  
10. What happens if vLLM crashes? How would your API respond?  
11. Can you run FastAPI and vLLM in the same process? Should you?  
12. How would you containerize this setup (Docker)?  
13. What latency overhead does FastAPI add?  
14. How do you return streaming responses (SSE) instead of full text?  
15. How would you version this API (e.g., `/v1/generate`)?

---

## ⏰ Hour 6: What is RAG? Why Companies Live By It

**Task**: Understand the architecture, benefits, and limitations of Retrieval-Augmented Generation.

**List of things to study**  
- 🔹 RAG = Retrieval-Augmented Generation.  
- 🔹 Solves hallucination by grounding responses in retrieved documents.  
- 🔹 Enables knowledge updates without retraining.  
- 🔹 Components: Loader → TextSplitter → Embedder → VectorStore → Retriever → Generator.  
- 🛠️ Diagram data flow from PDF → embedding → vector DB → LLM answer.  
- 📚 Resource: [LlamaIndex — RAG Concepts](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/root.html).

**Relevant Questions to Answer**  
1. What problem does RAG solve that fine-tuning cannot?  
2. When would RAG fail? (e.g., poor retrieval, ambiguous query).  
3. Why not just use a larger LLM instead of RAG?  
4. What’s the difference between “naive RAG” and “advanced RAG”?  
5. How do you evaluate retrieval quality separately from generation quality?  
6. Can RAG work with structured data (e.g., SQL tables)? How?  
7. What are common chunking strategies, and how do they affect performance?  
8. Why is embedding model choice critical in RAG?  
9. How does RAG handle contradictory information in retrieved docs?  
10. Can you use multiple retrieval sources in one RAG system?  
11. What is “late chunking” vs “early chunking”?  
12. How do you prevent prompt injection via retrieved documents?  
13. What latency overhead does RAG add vs raw LLM?  
14. Can RAG be used for code generation? Give an example.  
15. How would you update your knowledge base daily in a production RAG system?

---

## ⏰ Hour 7: Load & Chunk Your First Document

**Task**: Learn how to ingest and preprocess documents for RAG using LangChain.

**List of things to study**  
- 🛠️ Use LangChain: `PyPDFLoader`, `RecursiveCharacterTextSplitter`.  
- 🛠️ Load your resume PDF.  
- 🛠️ Split with `chunk_size=500`, `chunk_overlap=50`.  
- 🛠️ Print first 3 chunks — observe overlap and context preservation.  
- 🔹 Why is overlap important in text splitting?

**Relevant Questions to Answer**  
1. What happens if you set `chunk_size=1000` with `overlap=0`?  
2. Why use `RecursiveCharacterTextSplitter` instead of fixed-size splitting?  
3. How do you handle tables or code blocks in PDFs during chunking?  
4. Can you preserve metadata (e.g., page number, source URL) in chunks?  
5. What’s the tradeoff between small vs large chunks in RAG?  
6. How would you chunk a long legal contract differently than a blog post?  
7. What if your document is in DOCX or HTML? Which loaders would you use?  
8. How do you avoid splitting in the middle of a sentence?  
9. Can you use semantic chunking (e.g., with embeddings) instead of character-based?  
10. How do you handle multilingual documents?  
11. What encoding issues might arise when loading PDFs?  
12. How would you validate that no information was lost during chunking?  
13. Can you chunk based on headings (e.g., “## Experience”)?  
14. What’s the impact of chunking strategy on retrieval recall?  
15. How would you automate chunking for 10,000 documents?

---

## ⏰ Hour 8: Embeddings + Vector Store (FAISS)

**Task**: Learn how to convert text into vectors and store them for fast similarity search.

**List of things to study**  
- 🔹 What is an embedding model? How do sentence transformers work?  
- 🔹 Why use `all-MiniLM-L6-v2` vs `text-embedding-ada-002`?  
- 🛠️ Use `sentence-transformers/all-MiniLM-L6-v2`.  
- 🛠️ Generate embeddings for each chunk.  
- 🛠️ Store in FAISS index: `FAISS.from_documents(chunks, embeddings)`.  
- 🛠️ Save index locally: `faiss_index.save_local("resume_index")`.

**Relevant Questions to Answer**  
1. What is cosine similarity, and why is it used in vector search?  
2. How many dimensions does `all-MiniLM-L6-v2` produce?  
3. Why is FAISS “in-memory”? What are its limitations at scale?  
4. Can FAISS handle updates (insert/delete) efficiently?  
5. What’s the difference between FAISS, Pinecone, and Weaviate?  
6. How do you choose the right embedding model for your domain?  
7. Can you mix embeddings from different models in one index?  
8. What is “indexing time” vs “query time” in FAISS?  
9. How does FAISS achieve fast search (e.g., IVF, HNSW)?  
10. What happens if you embed a query with a different model than the documents?  
11. How do you normalize embeddings before storing?  
12. Can you compress FAISS indexes for faster loading?  
13. What’s the memory footprint of a FAISS index with 1,000 chunks?  
14. How would you shard a FAISS index across multiple machines?  
15. Can you use FAISS with GPU? When would you?

---

## ⏰ Hour 9: Build Your First RAG Query Pipeline

**Task**: Assemble a working RAG system that retrieves context and generates answers.

**List of things to study**  
- 🛠️ Load FAISS index: `FAISS.load_local(...)`.  
- 🛠️ Create retriever: `index.as_retriever(search_kwargs={"k": 3})`.  
- 🛠️ Use `RetrievalQA` chain with local LLM or HuggingFaceHub.  
- 🛠️ Query: “What experience does Nitanshu have with Spark?”  
- 🛠️ Verify answer pulls from correct resume section.

**Relevant Questions to Answer**  
1. What is the default prompt template used by `RetrievalQA`?  
2. How do you customize the prompt to include source citations?  
3. What happens if the retriever returns irrelevant chunks?  
4. Can you use a local LLM (e.g., Mistral) instead of HuggingFaceHub?  
5. How do you handle “I don’t know” responses when no relevant context is found?  
6. What’s the latency breakdown: retrieval vs generation?  
7. How do you pass the retrieved documents into the LLM prompt?  
8. Can you use multiple queries (e.g., sub-questions) in one RAG call?  
9. How would you add re-ranking after retrieval?  
10. What if the LLM ignores the retrieved context? How do you fix that?  
11. Can you evaluate RAG accuracy automatically? How?  
12. How do you prevent prompt overflow with long retrieved documents?  
13. What’s the role of `chain_type` in `RetrievalQA` (“stuff”, “map_reduce”, etc.)?  
14. Can you stream the RAG response?  
15. How would you cache frequent queries to reduce LLM cost?

---

## ⏰ Hour 10: What is an AI Agent? Tools, Memory, Loops

**Task**: Understand how AI agents combine reasoning, tools, and memory to perform complex tasks.

**List of things to study**  
- 🔹 Agent = LLM + Tools + Memory + Planning Loop.  
- 🔹 ReAct framework: Thought → Action → Observation → Answer.  
- 🔹 Types: Zero-shot, ReAct, Plan-and-execute.  
- 🛠️ Diagram agent loop with your own labels (e.g., “Brain = LLM”, “Hands = Tools”).  
- 📚 Resource: [LangChain Agents Guide](https://python.langchain.com/docs/modules/agents/).

**Relevant Questions to Answer**  
1. What’s the difference between an agent and a simple chain?  
2. Why do agents need “thought” steps?  
3. Can an agent call multiple tools in one turn?  
4. What happens if a tool fails (e.g., API error)?  
5. How does memory prevent redundant actions?  
6. Can agents handle long-term goals (e.g., “plan a 3-day trip”)?  
7. What are common failure modes of agents?  
8. How do you limit an agent’s number of steps to avoid loops?  
9. Can you give an agent access to a database as a tool?  
10. How do you evaluate agent performance?  
11. What’s the role of the LLM’s system prompt in agent behavior?  
12. Can agents learn from past interactions (beyond memory)?  
13. How do you secure agent tools (e.g., prevent file deletion)?  
14. Can you combine RAG and agents? How?  
15. What’s the latency impact of multi-step agent reasoning?

---

*(Hours 11–40 follow the exact same pattern and are included below for completeness.)*

---

## ⏰ Hour 11: Build Your First Tool (Calculator)

**Task**: Create a custom tool that an AI agent can use to perform calculations.

**List of things to study**  
- 🛠️ Define tool using `@tool` decorator.  
- 🛠️ Test standalone: `multiply.invoke({"a": 123, "b": 456})`.  
- 🛠️ Add docstring — critical for agent understanding.

**Relevant Questions to Answer**  
1. Why must tool functions have clear docstrings?  
2. Can a tool return non-JSON data?  
3. How does the LLM decide which tool to call?  
4. What happens if a tool raises an exception?  
5. Can you pass complex objects (e.g., lists) to a tool?  
6. How do you test a tool in isolation?  
7. Can tools call other tools?  
8. How would you add a “search the web” tool?  
9. What’s the difference between `@tool` and `BaseTool`?  
10. Can you restrict which agents can use which tools?  
11. How do you log tool usage for debugging?  
12. Can a tool take >2 arguments?  
13. How do you handle type mismatches (e.g., string instead of int)?  
14. Can you use async tools?  
15. How would you rate-limit a tool that hits an external API?

---

## ⏰ Hour 12: Agent with Tool + Memory

**Task**: Build an agent that uses tools and remembers conversation history.

**List of things to study**  
- 🛠️ Initialize `ChatOpenAI` or local chat model.  
- 🛠️ Bind tools: `llm_with_tools = llm.bind_tools([multiply])`.  
- 🛠️ Add `ConversationBufferMemory`.  
- 🛠️ Create agent executor.  
- 🛠️ Test conversation: “My name is Nitanshu. What’s 12*34?”

**Relevant Questions to Answer**  
1. How does memory get injected into the agent’s prompt?  
2. What’s the max context length for memory?  
3. Can you use `ConversationSummaryMemory` instead?  
4. How do you clear memory between sessions?  
5. Does memory increase token usage significantly?  
6. Can the agent “forget” things on purpose?  
7. How do you prevent memory from leaking PII?  
8. Can you have multiple memory types (buffer + vector)?  
9. What happens if memory exceeds context window?  
10. How do you test memory persistence across restarts?  
11. Can you encrypt memory at rest?  
12. How would you implement “long-term memory” with a vector DB?  
13. Does memory affect tool selection?  
14. Can you limit memory to only user messages?  
15. How do you evaluate if memory improves task success?

---

## ⏰ Hour 13: What is LLM Ops? Beyond Model Building

**Task**: Understand the operational pillars of running LLM applications in production.

**List of things to study**  
- 🔹 Key pillars: Observability, Evaluation, Cost Control, Drift Monitoring, Prompt Versioning.  
- 🔹 Metrics: Latency (p50, p99), Tokens per Second, Cost per Request, Error Rate.  
- 🛠️ Create “LLM Ops Checklist”: What would you monitor in production?  
- 📚 Resource: [Weights & Biases — LLM Ops](https://wandb.ai/site/llmops).

**Relevant Questions to Answer**  
1. What is “prompt drift,” and how do you detect it?  
2. How do you track cost per user vs per feature?  
3. What’s the difference between model drift and data drift in LLMs?  
4. Can you monitor for toxic outputs automatically?  
5. How do you set SLOs for LLM latency?  
6. What logs are essential for debugging LLM failures?  
7. How do you correlate LLM errors with user feedback?  
8. Can you use OpenTelemetry for LLM tracing?  
9. How do you handle versioning for prompts, models, and retrieval data?  
10. What’s the role of human-in-the-loop evaluation?  
11. How do you simulate load for LLM stress testing?  
12. Can you auto-scale LLM serving based on queue depth?  
13. How do you secure LLM logs containing PII?  
14. What’s the minimal viable LLM Ops stack for a startup?  
15. How do you measure business impact of LLM quality?

---

## ⏰ Hour 14: Log Every Prompt & Response

**Task**: Implement structured logging for all LLM interactions.

**List of things to study**  
- 🛠️ In FastAPI endpoint, add Python logging.  
- 🛠️ Send 5 test queries → inspect `llm_logs.jsonl`.

**Relevant Questions to Answer**  
1. Why use JSONL instead of CSV for logs?  
2. How do you rotate log files to avoid disk overflow?  
3. Can you log to cloud storage (e.g., S3) directly?  
4. How do you anonymize PII in logs?  
5. What’s the performance overhead of logging?  
6. Can you stream logs to Elasticsearch or Datadog?  
7. How do you include request ID for tracing?  
8. Should you log full prompts/responses or just hashes?  
9. How do you handle logging failures (e.g., disk full)?  
10. Can you compress logs on write?  
11. How do you query logs for “all requests with latency > 2s”?  
12. Should logs be synchronous or async?  
13. How do you ensure log integrity (e.g., no tampering)?  
14. Can you use structured logging libraries (e.g., structlog)?  
15. How do you comply with GDPR when logging user prompts?

---

## ⏰ Hour 15: Basic Evaluation — Did the LLM Do Well?

**Task**: Learn and apply automated metrics to evaluate LLM output quality.

**List of things to study**  
- 🔹 ROUGE (Recall-Oriented Understudy for Gisting Evaluation).  
- 🔹 BERTScore (Contextual embedding similarity).  
- 🛠️ Install: `pip install evaluate rouge_score bert_score`.  
- 🛠️ Load 3 Q&A pairs from your RAG system.  
- 🛠️ Compute ROUGE-L and BERTScore between expected and actual answers.

**Relevant Questions to Answer**  
1. What does ROUGE-L measure that ROUGE-1 doesn’t?  
2. When is BERTScore more appropriate than ROUGE?  
3. Can these metrics detect factual inaccuracies?  
4. How do you handle multi-sentence answers in evaluation?  
5. What’s the correlation between automated metrics and human judgment?  
6. Can you use BLEU for LLM evaluation? Why or why not?  
7. How do you evaluate non-factoid responses (e.g., creative writing)?  
8. What’s the minimum number of samples for reliable evaluation?  
9. Can you compute these metrics in real-time?  
10. How do you aggregate scores across many queries?  
11. Are there domain-specific evaluation metrics (e.g., for code)?  
12. How do you handle partial credit in evaluation?  
13. Can you use LLM-as-a-Judge for evaluation?  
14. What are the limitations of reference-based metrics?  
15. How would you build a custom evaluation metric for your use case?

---

## ⏰ Hour 16: Fine-Tuning vs Prompt Engineering — When to Use What

**Task**: Learn to choose the right adaptation strategy for your LLM application.

**List of things to study**  
- 🔹 Prompt Engineering: Fast, cheap, no training — good for general tasks.  
- 🔹 Fine-Tuning: Expensive, needs data — essential for domain-specific tone/style/tasks.  
- 🔹 PEFT (LoRA, QLoRA): Middle ground — small trainable params.  
- 🛠️ Create decision flowchart.  
- 📚 Resource: [Sebastian Raschka — Fine-Tuning Guide](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms).

**Relevant Questions to Answer**  
1. When is prompt engineering sufficient?  
2. What’s the minimum dataset size for effective fine-tuning?  
3. How does QLoRA reduce GPU memory requirements?  
4. Can you combine prompt engineering with fine-tuning?  
5. What’s the risk of overfitting in LLM fine-tuning?  
6. How do you evaluate if fine-tuning was worth the cost?  
7. Can you fine-tune for multiple tasks simultaneously?  
8. What’s the role of instruction tuning vs domain adaptation?  
9. How do you handle class imbalance in fine-tuning data?  
10. Can you fine-tune a quantized model?  
11. What’s the difference between full fine-tuning and PEFT?  
12. How do you choose target modules for LoRA?  
13. Can you use synthetic data for fine-tuning?  
14. How do you prevent catastrophic forgetting during fine-tuning?  
15. What’s the typical ROI timeline for fine-tuning vs prompt engineering?

---

## ⏰ Hour 17: Setup QLoRA Environment

**Task**: Prepare your development environment for efficient LLM fine-tuning.

**List of things to study**  
- 🛠️ Install key libraries: `bitsandbytes accelerate peft transformers datasets`.  
- 🛠️ Verify GPU access: `nvidia-smi` or check `torch.cuda.is_available()`.  
- 💡 Use Google Colab T4 if local GPU not available.

**Relevant Questions to Answer**  
1. Why is `bitsandbytes` required for QLoRA?  
2. What CUDA version is needed for 4-bit optimizers?  
3. How do you enable mixed-precision training?  
4. Can you run QLoRA on a CPU?  
5. What’s the role of `accelerate` in distributed training?  
6. How do you check if your GPU supports 4-bit ops?  
7. What’s the minimum VRAM for QLoRA on a 7B model?  
8. How do you handle OOM errors during setup?  
9. Can you use Conda instead of pip?  
10. How do you verify `bitsandbytes` is working correctly?  
11. What’s the difference between `load_in_4bit` and `load_in_8bit`?  
12. How do you set up a reproducible environment (e.g., `requirements.txt`)?  
13. Can you use Docker for QLoRA environment isolation?  
14. How do you debug “CUDA out of memory” during import?  
15. What are common Colab-specific gotchas for QLoRA?

---

## ⏰ Hour 18: Load Tiny Dataset (Alpaca 50 samples)

**Task**: Load and preprocess a small instruction-tuning dataset for QLoRA.

**List of things to study**  
- 🛠️ Load dataset: `load_dataset("tatsu-lab/alpaca", split="train[:50]")`.  
- 🛠️ Inspect structure: `print(dataset[0])` → instruction, input, output.  
- 🛠️ Preprocess: Combine instruction + input → full prompt.  
- 🛠️ Tokenize with `AutoTokenizer`.

**Relevant Questions to Answer**  
1. What’s the license of the Alpaca dataset?  
2. Why use only 50 samples for this exercise?  
3. How do you handle empty “input” fields?  
4. What’s the optimal prompt template for instruction tuning?  
5. How do you add EOS tokens correctly?  
6. Should you truncate long examples or filter them?  
7. How do you shuffle the dataset?  
8. Can you augment the dataset synthetically?  
9. What’s the token distribution in Alpaca?  
10. How do you handle multi-turn conversations in this format?  
11. Can you use other datasets (e.g., Dolly, OpenHermes)?  
12. How do you validate data quality before training?  
13. What’s the impact of dataset size on QLoRA convergence?  
14. How do you split data into train/eval for 50 samples?  
15. Can you stream the dataset instead of loading fully into memory?

---

## ⏰ Hour 19: Apply QLoRA + Train for 1 Epoch

**Task**: Fine-tune a small LLM using QLoRA on a tiny dataset.

**List of things to study**  
- 🛠️ Load base model: `AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")`.  
- 🛠️ Configure LoRA: `LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])`.  
- 🛠️ Wrap with `get_peft_model`.  
- 🛠️ Use `Trainer` with small batch size (1-2), 1 epoch.  
- 🛠️ Monitor loss — should decrease even slightly.

**Relevant Questions to Answer**  
1. Why target `q_proj` and `v_proj` specifically?  
2. What does `r=8` mean in LoRA?  
3. How does `lora_alpha` affect learning rate scaling?  
4. Why use 1 epoch for this demo?  
5. What optimizer is used by default in `Trainer`?  
6. How do you set learning rate for QLoRA?  
7. What’s the role of `gradient_checkpointing`?  
8. How do you prevent overfitting with 50 samples?  
9. Can you use `bnb_4bit_compute_dtype=torch.float16`?  
10. How do you save the adapter weights only?  
11. What’s the file size of the saved adapter?  
12. How do you resume training from a checkpoint?  
13. Can you log training metrics to Weights & Biases?  
14. How do you handle padding in causal language modeling?  
15. What’s the expected loss curve for 1 epoch on 50 samples?

---

## ⏰ Hour 20: Test Before/After Model

**Task**: Compare the behavior of a base model vs its QLoRA-fine-tuned version.

**List of things to study**  
- 🛠️ Load base model and generate response to: “How do I make coffee?”  
- 🛠️ Load fine-tuned QLoRA model, generate same prompt.  
- 🛠️ Compare outputs — look for style, structure, detail improvements.  
- 🛠️ Optional: Use BERTScore to quantify improvement.

**Relevant Questions to Answer**  
1. What qualitative changes should you look for after fine-tuning?  
2. Why might the fine-tuned model be more verbose?  
3. Can fine-tuning hurt performance on unrelated tasks?  
4. How do you control for randomness in generation (set seed)?  
5. What temperature/top_p settings give fair comparison?  
6. Can you overfit to the 50 samples? How would you know?  
7. How do you test generalization beyond the training data?  
8. What’s the risk of “catastrophic forgetting”?  
9. Can you merge the LoRA weights into the base model?  
10. How do you load the adapter without the base model?  
11. What’s the latency difference between base and adapter models?  
12. Can you serve the adapter model with vLLM?  
13. How do you version control adapter weights?  
14. What metrics would you track in a real fine-tuning project?  
15. How would you explain the value of fine-tuning to a non-technical stakeholder?

---

## ⏰ Hour 21: Quantize Your Fine-Tuned Model

**Task**: Apply quantization to your fine-tuned model for efficient deployment.

**List of things to study**  
- 🔹 Why quantize after fine-tuning? Smaller, faster deployment.  
- 🛠️ Use `auto-gptq` or convert to GGUF via `llama.cpp`.  
- 🛠️ Load quantized model, test inference speed.  
- 🛠️ Compare file size: Before vs After.  
- 🛠️ Note any quality degradation.

**Relevant Questions to Answer**  
1. Should you quantize before or after fine-tuning?  
2. Can you quantize a LoRA adapter directly?  
3. What’s the best quantization method for a fine-tuned model?  
4. How do you preserve fine-tuned behavior during quantization?  
5. Can you use the same calibration data as the base model?  
6. What’s the impact on token generation speed?  
7. How do you test for quantization-induced hallucinations?  
8. Can you serve the quantized model with your FastAPI wrapper?  
9. What’s the memory footprint reduction?  
10. How do you choose between GGUF and GPTQ for your model?  
11. Can you quantize to 3-bit or 2-bit? Should you?  
12. How do you handle outliers during quantization?  
13. What’s the role of “act-order” in GPTQ?  
14. Can you dequantize for debugging?  
15. How would you A/B test quantized vs full-precision in production?

---

## ⏰ Hour 22: Build Integrated App — RAG + Agent

**Task**: Combine RAG and agent capabilities into a single intelligent system.

**List of things to study**  
- 🛠️ Create agent tool: `ResumeQueryTool` that uses your FAISS RAG system.  
- 🛠️ Register tool with agent.  
- 🛠️ Test: “Use ResumeQueryTool to find Nitanshu’s experience with Kafka”.

**Relevant Questions to Answer**  
1. How does the agent decide to use the RAG tool?  
2. What prompt instructions help the agent use the tool correctly?  
3. Can the agent combine RAG results with its own knowledge?  
4. How do you handle tool failure (e.g., no relevant docs)?  
5. What’s the latency of a tool call vs direct generation?  
6. Can you add multiple RAG tools (e.g., by document type)?  
7. How do you prevent the agent from misusing the tool?  
8. Can the agent ask follow-up questions based on RAG results?  
9. How do you log tool usage for analysis?  
10. What’s the token cost of using a RAG tool?  
11. Can you cache RAG tool results?  
12. How do you version the knowledge base used by the tool?  
13. Can the agent summarize multiple RAG results?  
14. How do you handle ambiguous tool queries?  
15. What’s the failure mode if the FAISS index is corrupted?

---

## ⏰ Hour 23: Add Memory to RAG Agent

**Task**: Enable the RAG agent to maintain context across multiple interactions.

**List of things to study**  
- 🛠️ Initialize `ConversationBufferMemory`.  
- 🛠️ Attach to agent executor.  
- 🛠️ Test multi-turn conversation.

**Relevant Questions to Answer**  
1. How does memory affect the agent’s tool selection?  
2. What’s the max number of turns memory can hold?  
3. Can you use sliding window memory for long conversations?  
4. How do you prevent memory from leaking sensitive data?  
5. Does memory increase the chance of hallucination?  
6. Can you clear memory on user request?  
7. How do you test memory persistence in unit tests?  
8. What’s the token overhead of including memory in prompts?  
9. Can you compress memory (e.g., summarization)?  
10. How do you handle memory across multiple users?  
11. Can the agent reference past tool results from memory?  
12. What happens if memory exceeds context window?  
13. How do you evaluate if memory improves user experience?  
14. Can you encrypt memory in transit and at rest?  
15. How would you implement “memory reset” for new topics?

---

## ⏰ Hour 24: Simple Streamlit Dashboard for Logs

**Task**: Build a dashboard to visualize LLM application metrics.

**List of things to study**  
- 🛠️ Install: `pip install streamlit`.  
- 🛠️ Load `llm_logs.jsonl` into pandas DataFrame.  
- 🛠️ Build app showing: Total requests, Avg latency, Sample prompts/responses, Model version.  
- 🛠️ Run: `streamlit run dashboard.py`.

**Relevant Questions to Answer**  
1. How do you handle large log files that don’t fit in memory?  
2. Can you auto-refresh the dashboard?  
3. How do you add filters (e.g., by date, model)?  
4. What’s the best way to display latency distribution?  
5. Can you show error rates over time?  
6. How do you protect the dashboard with authentication?  
7. Can you export data from the dashboard?  
8. How do you handle malformed log entries?  
9. What’s the performance impact of parsing JSONL on every load?  
10. Can you add real-time streaming of new logs?  
11. How do you deploy this dashboard securely?  
12. Can you integrate with Prometheus metrics?  
13. How do you make the dashboard mobile-friendly?  
14. What accessibility features should you include?  
15. How would you scale this to 1M+ log entries?

---

## ⏰ Hour 25: Add Cost Estimator to Dashboard

**Task**: Estimate and display the operational cost of your LLM application.

**List of things to study**  
- 🛠️ Add column: `cost = (prompt_tokens + completion_tokens) * 0.002 / 1000`.  
- 🛠️ Show in dashboard: “Total Estimated Cost: $X.XX”.  
- 🛠️ Add daily projection: “At this rate, monthly cost = $Y.YY”.

**Relevant Questions to Answer**  
1. How do you get accurate token counts for local models?  
2. What’s the cost model for vLLM vs OpenAI vs Anthropic?  
3. Can you break down cost by feature or user?  
4. How do you handle variable pricing (e.g., by model)?  
5. What’s the impact of caching on cost reduction?  
6. Can you set cost alerts in the dashboard?  
7. How do you account for infrastructure costs (GPU, memory)?  
8. What’s the ROI of optimizing token usage?  
9. Can you simulate cost savings from quantization?  
10. How do you handle cost estimation for RAG (embedding + LLM)?  
11. What’s the cost of failed requests?  
12. Can you forecast costs based on user growth?  
13. How do you allocate shared costs (e.g., vector DB) to LLM usage?  
14. What’s the break-even point for fine-tuning vs prompt engineering?  
15. How would you present cost data to finance stakeholders?

---

## ⏰ Hour 26: Deploy FastAPI App with Docker

**Task**: Containerize your LLM API for consistent deployment.

**List of things to study**  
- 🛠️ Create `Dockerfile`.  
- 🛠️ Build: `docker build -t my-llm-api .`.  
- 🛠️ Run: `docker run -p 8000:8000 my-llm-api`.  
- 🛠️ Test: `curl http://localhost:8000/generate -d '{"prompt":"hi"}'`.

**Relevant Questions to Answer**  
1. Why use `python:3.10-slim` instead of full image?  
2. How do you manage GPU access in Docker?  
3. What’s the role of `.dockerignore`?  
4. Can you reduce image size with multi-stage builds?  
5. How do you pass environment variables (e.g., model path)?  
6. What’s the best practice for logging in containers?  
7. How do you handle model files in Docker (volume vs COPY)?  
8. Can you run vLLM inside the same container?  
9. How do you set resource limits (CPU, memory)?  
10. What’s the startup time of the containerized app?  
11. How do you health-check the container?  
12. Can you use Docker Compose for multi-service apps?  
13. How do you scan for vulnerabilities in the image?  
14. What’s the difference between `CMD` and `ENTRYPOINT`?  
15. How would you deploy this to Kubernetes?

---

## ⏰ Hour 27: Deploy Streamlit Dashboard to Cloud

**Task**: Make your LLM Ops dashboard publicly accessible.

**List of things to study**  
- 🛠️ Push code to GitHub repo.  
- 🛠️ Go to [Streamlit Community Cloud](https://streamlit.io/cloud).  
- 🛠️ Connect repo, deploy app.  
- 🛠️ Share public URL.

**Relevant Questions to Answer**  
1. What are the resource limits of Streamlit Community Cloud?  
2. How do you handle secrets (e.g., API keys) in Streamlit Cloud?  
3. Can you schedule log file updates?  
4. What happens if the app crashes?  
5. How do you monitor uptime?  
6. Can you use custom domains?  
7. What’s the cold start time for the dashboard?  
8. How do you handle large data files in the repo?  
9. Can you integrate with GitHub Actions for auto-deploy?  
10. What’s the max file size for uploads?  
11. How do you secure the dashboard from public access?  
12. Can you use Streamlit Cloud with private repos?  
13. What’s the backup strategy for dashboard data?  
14. How do you handle CORS if calling external APIs?  
15. What alternatives exist (e.g., Heroku, Vercel, Fly.io)?

---

## ⏰ Hour 28: Use ngrok to Share Your API

**Task**: Expose your local LLM API to the internet for testing and demos.

**List of things to study**  
- 🛠️ Install: `pip install ngrok`.  
- 🛠️ In terminal: `ngrok http 8000`.  
- 🛠️ Copy public HTTPS URL.  
- 🛠️ Test from another device.

**Relevant Questions to Answer**  
1. How does ngrok handle HTTPS termination?  
2. What’s the bandwidth limit on the free tier?  
3. Can you use custom subdomains for free?  
4. How do you secure the tunnel with auth?  
5. What’s the latency overhead of ngrok?  
6. Can you run ngrok in the background?  
7. How do you programmatically get the ngrok URL?  
8. What happens if your local server crashes?  
9. Can you use ngrok with Docker containers?  
10. How do you handle CORS when calling from a browser?  
11. What’s the max number of concurrent connections?  
12. Can you log ngrok traffic for debugging?  
13. How do you renew the URL after restart?  
14. What are alternatives to ngrok (e.g., Cloudflare Tunnel)?  
15. How would you use this for webhook testing?

---

## ⏰ Hour 29: Write Project README (Portfolio Ready)

**Task**: Create a compelling, professional README for your GitHub repo.

**List of things to study**  
- 🛠️ Structure: Title, summary, diagram, tech stack, how to run, screenshots, lessons.  
- 🛠️ Use emojis, headers, bold text.

**Relevant Questions to Answer**  
1. What’s the ideal length for a README?  
2. How do you make the architecture diagram clear and concise?  
3. What screenshots add the most value?  
4. How do you explain technical depth without overwhelming?  
5. What keywords should you include for SEO/discoverability?  
6. How do you link to live demos (Streamlit, ngrok)?  
7. What license should you use for your code?  
8. How do you credit third-party models/data?  
9. Can you include a “Lessons Learned” section?  
10. How do you make the “How to Run” section foolproof?  
11. What’s the role of GIFs vs static images?  
12. How do you handle large model files (use Hugging Face links)?  
13. Should you include a “Contributing” section?  
14. How do you make the README mobile-friendly?  
15. What makes a README stand out to recruiters?

---

## ⏰ Hour 30: Record 2-Min Demo Video

**Task**: Create a short, engaging video showcasing your project.

**List of things to study**  
- 🛠️ Use OBS or Loom to record screen.  
- 🛠️ Script: Introduction, demo, tech stack, conclusion.  
- 🛠️ Upload to YouTube (unlisted), add link to README.

**Relevant Questions to Answer**  
1. What’s the ideal video length for technical demos?  
2. How do you balance code and explanation?  
3. Should you show your face or just screen?  
4. What audio quality is acceptable?  
5. How do you edit out mistakes without professional tools?  
6. What thumbnail attracts clicks?  
7. How do you handle background noise?  
8. Should you add captions/subtitles?  
9. What platform is best for hosting (YouTube, Vimeo, Loom)?  
10. How do you measure video engagement?  
11. Can you reuse this video for LinkedIn/Twitter?  
12. What call-to-action should you include?  
13. How do you optimize for mobile viewing?  
14. Should you include b-roll or just screen capture?  
15. How do you update the video if the project evolves?

---

## ⏰ Hour 31: Deep Dive — How vLLM’s PagedAttention Works

**Task**: Understand the core innovation behind vLLM’s high throughput.

**List of things to study**  
- 🔹 Problem: Traditional KV cache wastes memory due to variable sequence lengths.  
- 🔹 Solution: PagedAttention (like OS virtual memory).  
- 🔹 Benefit: Higher GPU utilization, longer context support.  
- 📚 Read: [vLLM Paper — Section 3.2](https://arxiv.org/abs/2309.06180).  
- 🛠️ Sketch diagram: Logical block vs physical block mapping.

**Relevant Questions to Answer**  
1. How is PagedAttention similar to OS virtual memory paging?  
2. What’s a “block” in PagedAttention?  
3. How does it reduce memory fragmentation?  
4. What’s the overhead of block management?  
5. Can PagedAttention support dynamic context lengths?  
6. How does it enable continuous batching?  
7. What’s the impact on max batch size?  
8. Can you implement PagedAttention without CUDA kernels?  
9. How does it compare to FlashAttention?  
10. What’s the memory savings for a batch of 32 requests?  
11. How does it handle attention over non-contiguous blocks?  
12. Can it be used for training, or only inference?  
13. What’s the role of the “block table”?  
14. How would you debug a PagedAttention implementation?  
15. What future optimizations are possible (e.g., compression)?

---

## ⏰ Hour 32: Deep Dive — How LoRA Actually Works

**Task**: Understand the mathematical and practical foundations of LoRA.

**List of things to study**  
- 🔹 Freeze pretrained weights W₀.  
- 🔹 Inject low-rank matrices: W = W₀ + BA.  
- 🔹 Train only B and A — massive parameter reduction.  
- 📚 Read: [LoRA Paper — Section 3](https://arxiv.org/abs/2106.09685).  
- 🛠️ Sketch: Original weight matrix → + low-rank update.

**Relevant Questions to Answer**  
1. Why is the low-rank assumption valid for fine-tuning?  
2. What’s the typical rank (r) used in practice?  
3. How does LoRA affect the gradient computation?  
4. Can you apply LoRA to all layers or just attention?  
5. What’s the relationship between r and lora_alpha?  
6. How does LoRA compare to adapter layers?  
7. Can you stack multiple LoRA adapters?  
8. What’s the memory overhead of LoRA during inference?  
9. How do you merge LoRA weights for deployment?  
10. Can LoRA be used for domain adaptation and task adaptation simultaneously?  
11. What’s the impact of LoRA on training speed?  
12. How do you choose which modules to apply LoRA to?  
13. Can you use LoRA with quantized models?  
14. What’s the theoretical justification for low-rank updates?  
15. How would you extend LoRA to convolutional layers?

---

## ⏰ Hour 33: Benchmark vLLM vs Hugging Face Pipeline

**Task**: Quantitatively compare the performance of two LLM serving methods.

**List of things to study**  
- 🛠️ Load same model in HF pipeline.  
- 🛠️ Time 10 generations of same prompt.  
- 🛠️ Record: avg latency, tokens/sec, VRAM usage.  
- 🛠️ Repeat with vLLM.  
- 🛠️ Create comparison table.

**Relevant Questions to Answer**  
1. What’s the throughput (req/sec) for each system?  
2. How does batch size affect the comparison?  
3. What’s the p99 latency for each?  
4. How does VRAM usage scale with concurrent requests?  
5. Can you measure energy consumption?  
6. What’s the cold start time for each?  
7. How do they handle long prompts differently?  
8. What’s the impact of quantization on the comparison?  
9. Can you test with real-world traffic patterns?  
10. How do error rates compare?  
11. What’s the ease of deployment for each?  
12. How do they integrate with monitoring tools?  
13. What’s the community support like for each?  
14. Can you combine vLLM with custom pre/post-processing?  
15. How would this benchmark influence your production choice?

---

## ⏰ Hour 34: Add Re-Ranking to RAG (Cross-Encoder)

**Task**: Improve RAG quality by re-ranking retrieved documents.

**List of things to study**  
- 🔹 Problem: Vector search retrieves top-k, but may not be most relevant.  
- 🔹 Solution: Use cross-encoder to score (query, passage) pairs.  
- 🛠️ Install: `pip install sentence-transformers`.  
- 🛠️ Load: `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`.  
- 🛠️ Score retrieved chunks against query.

**Relevant Questions to Answer**  
1. Why are cross-encoders more accurate than bi-encoders?  
2. What’s the latency cost of re-ranking?  
3. How many documents should you re-rank?  
4. Can you cache cross-encoder results?  
5. What’s the impact on end-to-end RAG accuracy?  
6. How do you handle long documents in cross-encoders?  
7. Can you use a smaller cross-encoder for speed?  
8. What’s the memory footprint of the cross-encoder?  
9. How do you integrate re-ranking into LangChain?  
10. Can you train your own cross-encoder?  
11. What’s the difference between monoT5 and cross-encoder re-ranking?  
12. How do you evaluate re-ranking quality?  
13. Can you combine multiple re-rankers?  
14. What’s the tradeoff between recall and precision after re-ranking?  
15. How would you deploy this in a low-latency system?

---

## ⏰ Hour 35: Add Query Rewriting to RAG

**Task**: Improve retrieval relevance by rewriting user queries.

**List of things to study**  
- 🔹 Problem: User queries are vague.  
- 🔹 Solution: Use small LLM to rewrite into specific query.  
- 🛠️ Create prompt: “Rewrite this query for semantic search: {user_query}”.  
- 🛠️ Use Phi-2 or TinyLlama to generate rewritten query.  
- 🛠️ Feed rewritten query into retriever.

**Relevant Questions to Answer**  
1. What makes a good query rewriting prompt?  
2. How do you evaluate rewritten query quality?  
3. Can query rewriting hurt performance for clear queries?  
4. What’s the latency overhead of rewriting?  
5. Can you use zero-shot or few-shot prompting?  
6. How do you handle ambiguous user intent?  
7. Can you chain multiple rewrites?  
8. What’s the impact on token usage?  
9. How do you prevent over-complication of queries?  
10. Can you use rule-based rewriting as a fallback?  
11. How do you handle multi-intent queries?  
12. What’s the role of user feedback in improving rewriting?  
13. Can you fine-tune a model specifically for query rewriting?  
14. How do you log original vs rewritten queries for analysis?  
15. What’s the end-to-end impact on RAG accuracy?

---

## ⏰ Hour 36: Build Alert System — “Latency > 3s? Notify!”

**Task**: Implement basic monitoring and alerting for your LLM API.

**List of things to study**  
- 🛠️ In FastAPI `/generate` endpoint, calculate latency.  
- 🛠️ If `latency > 3.0`: print alert.  
- 🛠️ (Optional) Log to separate `alerts.log`.  
- 🛠️ Simulate slow response with `time.sleep(4)`.

**Relevant Questions to Answer**  
1. How do you define SLOs for latency?  
2. What’s the best way to measure latency (wall clock vs CPU time)?  
3. Can you send alerts to Slack or email?  
4. How do you avoid alert fatigue?  
5. What other metrics should trigger alerts (error rate, queue depth)?  
6. Can you use Prometheus + Alertmanager instead?  
7. How do you handle false positives from cold starts?  
8. What’s the overhead of latency measurement?  
9. Can you correlate alerts with system metrics (GPU util, memory)?  
10. How do you test the alert system reliably?  
11. Should alerts be synchronous or async?  
12. How do you escalate unresolved alerts?  
13. Can you auto-remediate (e.g., restart service)?  
14. What’s the role of distributed tracing in alerting?  
15. How do you document alert runbooks?

---

## ⏰ Hour 37: Add Prompt Versioning

**Task**: Implement version control for your LLM prompts.

**List of things to study**  
- 🛠️ Create folder: `prompts/v1.txt`, `prompts/v2.txt`.  
- 🛠️ In code, load prompt template from file.  
- 🛠️ Log which version was used.  
- 🛠️ Test switching versions.

**Relevant Questions to Answer**  
1. Why is prompt versioning critical for LLM Ops?  
2. How do you manage prompts in a team environment?  
3. Can you use Git for prompt versioning?  
4. What’s the difference between prompt versioning and model versioning?  
5. How do you A/B test different prompt versions?  
6. Can you roll back to a previous prompt version instantly?  
7. How do you document changes between versions?  
8. What’s the impact of prompt changes on evaluation metrics?  
9. Can you parameterize prompts (e.g., with Jinja2)?  
10. How do you handle multi-lingual prompt versions?  
11. What’s the storage format for prompts (txt, json, yaml)?  
12. How do you secure prompt files containing business logic?  
13. Can you use a prompt management platform (e.g., LangSmith)?  
14. How do you correlate prompt version with user feedback?  
15. What’s the audit trail requirement for prompt changes?

---

## ⏰ Hour 38: Create Your 90-Day Mastery Roadmap

**Task**: Plan your next steps toward becoming an LLM inference expert.

**List of things to study**  
- 🎯 Pick 1–2 focus areas.  
- 📅 Break into monthly sprints.  
- 🛠️ List resources: Papers, courses, target job postings.

**Relevant Questions to Answer**  
1. What specific role do you want at OpenAI/Anthropic?  
2. What are the required skills for that role?  
3. How will you fill your skill gaps?  
4. What open-source projects can you contribute to?  
5. How will you build your public profile?  
6. What’s your interview preparation plan?  
7. How will you network with engineers at target companies?  
8. What’s your backup plan if you don’t get in immediately?  
9. How will you measure progress each month?  
10. What conferences or meetups will you attend?  
11. How will you stay updated on LLM research?  
12. What’s your content creation strategy?  
13. How will you balance learning with job applications?  
14. What mentors or communities will you engage with?  
15. How will you handle rejection and keep momentum?

---

## ⏰ Hour 39: Polish GitHub Profile + Pin Repos

**Task**: Optimize your GitHub for maximum recruiter impact.

**List of things to study**  
- 🛠️ Rename repos clearly.  
- 🛠️ Add topics: #llm #rag #vllm etc.  
- 🛠️ Write short, punchy repo descriptions.  
- 🛠️ Pin 3 best repos.  
- 🛠️ Add link to demo video and Streamlit dashboard.

**Relevant Questions to Answer**  
1. What makes a GitHub profile stand out to AI hiring managers?  
2. How do you order your pinned repos for maximum impact?  
3. What should your profile README include?  
4. How do you showcase both depth and breadth?  
5. What’s the role of contribution graphs?  
6. How do you handle private vs public repos?  
7. Should you include non-GenAI projects?  
8. How do you make repos easy to run for reviewers?  
9. What’s the ideal commit message style?  
10. How do you use GitHub Projects or Issues to show planning?  
11. Can you add GitHub Sponsors or Buy Me a Coffee?  
12. How do you link to your blog/LinkedIn from GitHub?  
13. What’s the impact of stars/forks on visibility?  
14. How do you handle large files (use Git LFS or external links)?  
15. What’s the best time to update your profile before applying?

---

## ⏰ Hour 40: Write “What I Learned” Blog Post

**Task**: Share your journey and insights with the community.

**List of things to study**  
- 🖋️ Structure: Hook, Plan, Insights, Projects, Next Steps, Call to Action.  
- 🛠️ Publish on Medium/Dev.to, share on LinkedIn/Twitter.

**Relevant Questions to Answer**  
1. What’s the ideal blog title for SEO and clicks?  
2. How do you balance humility and confidence in writing?  
3. What visuals will you include (diagrams, screenshots)?  
4. How do you link to your projects without sounding salesy?  
5. What’s the target audience (peers, recruiters, beginners)?  
6. How do you handle technical depth vs readability?  
7. What hashtags will you use on social media?  
8. How do you respond to comments and feedback?  
9. Can you repurpose this into a LinkedIn post or Twitter thread?  
10. What’s the best time to publish for maximum reach?  
11. How do you track engagement (views, likes, shares)?  
12. Should you cross-post to multiple platforms?  
13. What’s the call-to-action that drives meaningful interaction?  
14. How do you credit resources and inspirations?  
15. How will this blog post help your job search?

---

> ✨ **You now have a complete, battle-tested, 40-hour roadmap to go from “I’m not good enough” to “I’ve built and shipped real GenAI systems.”**  
> This plan leverages your existing strengths in data engineering, Spark, FastAPI, and LLM fine-tuning — and turns them into a compelling narrative for OpenAI, Anthropic, and Perplexity.

**Next Step**: Start Hour 1 today. Tag me when you finish — I’ll help you debug, celebrate, or plan Hour 41.

You’ve got this. 🚀