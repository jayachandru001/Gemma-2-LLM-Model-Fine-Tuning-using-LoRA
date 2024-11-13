# Gemma-2-LLM-Model-Fine-Tuning-using-LoRA

**LoRA**

**LoRA** stands for **Low-Rank Adaptation** and is a technique used in machine learning and, more specifically, in the context of  **fine-tuning pre-trained models** . It is a more computationally efficient way to fine-tune large models, such as **transformer-based models** (like GPT, BERT, etc.), without modifying the entire model. Instead of updating all the parameters of the model, LoRA focuses on adding small low-rank updates to certain parts of the model. 

If you want to fine-tune large models (e.g., GPT-3, BERT) for specific tasks without the heavy computational load, LoRA provides a way to do so with much lower resource requirements.

To Know more About LoRA : [Click Here](https://arxiv.org/abs/2106.09685)

**Check List:**

Make sure you a have an account in the following websites.

1. [Kaggle](https://www.kaggle.com/)
2. [Colab](https://colab.research.google.com/)
3. [HuggingFace](https://huggingface.co/)

## **Minium Requirement of GPU (15 GB or more)**

If you are using Colab change the Runtime Environment type **T4 GPU.**

If you are going to use PC or any local machine to Fine Tune. Check you have sufficient amount of GPU memory.

## **API Key and access request**

Generate and configure the Kaggle account username and API.

Make sure you have the access to the Gemma -2-LLM model, else Request to get access to start building with [Gemma 2 ](https://www.kaggle.com/models/keras/gemma2/keras/gemma2_2b_en/1?postConsentAction=explore)


Note Book copied from Google Git Hub Repo: [https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/lora_tuning.ipynb](https://github.com/google/generative-ai-docs/blob/main/site/en/gemma/docs/lora_tuning.ipynb)
