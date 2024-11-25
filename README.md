### Scientific question-answering by fine-tuned large language models (LLMs) 

### Create an environment using the following command:
`conda env create --file env.yml`

### Dataset collection
- Galaxy help forum
- Biostars Q&A

### Fine-tune Llama2 (2B and 7B)
- Navigate to `\llama2` and then execute `python qlora-train.py`
- Utilizes HuggingFace's Transformers package to download pre-trained LLMs
- qLoRA to drastically reduce the number of parameters (from 2B to 6 million)
- SFT for setting up the training process

### Outcomes

![llama2_ans1](https://github.com/user-attachments/assets/74d92c7a-7522-4e58-8c33-4c3e418a8c93)

![llama2_answers](https://github.com/user-attachments/assets/1f368b2e-b2d3-436e-8888-2e10eb3e3622)

### Retrieval augmented generation (RAG)
- Dataset collection from Galaxy's training material and GitHub pull requests
- https://github.com/uwwint/discourse-scraper/blob/master/extract_documents/create_RAG_docs.ipynb
- Pipeline for RAG that uses fine-tuned Llama2 using Haystack

![rag_llm2](https://github.com/user-attachments/assets/d2458cae-e846-4da8-9542-78d69fd84a57)

### Save the fine-tuned model to HuggingFace Hub
- Save model to HuggingFace Hub: https://github.com/uwwint/discourse-scraper/blob/master/llama2/save_to_hub.ipynb
- Model name: anuprulez/fine-tuned-gllm 

