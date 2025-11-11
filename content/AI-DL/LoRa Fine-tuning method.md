---
title: LoRa Fine-tuning method
draft: false
tags:
  - AI
  - DL/AI
  - Project
---
This week, I decided to fine-tune the TinyLlama-1.1B model on a medical dataset. The goal is to provide the model with a specific scenario in which a person describes their situation and symptoms, and at the end, provides a list of possible options or diagnoses. The model is then expected to choose the most appropriate option and explain its reasoning.
## **Introduction** ##

The **LoRA** method allows us to fine-tune models while using significantly fewer parameters, which in turn reduces memory usage and GPU requirements.

For example, consider **BERT-Base**. BERT-Base is a language model that only uses the encoder part of the Transformer architecture. It consists of **12 encoder layers** (not 6), with a total of **110 million trainable parameters**. During standard fine-tuning, we need to store the gradients for all parameters, effectively requiring memory for at least **2 × 110 million parameters**.


# 1)Method 1: PEFT (Parameter efficient fine-tuning)

1) We use a different version of the **Encoder** architecture coming from the **"Parameter efficient Transfer Learning for NLP** paper

2) In this architecture we add a new Layer : **The Adapter Layer** ,  a layer that allows us  to use the **Parameter efficient fine-tuning (PEFT)** method. He is added on the top of every Feed-forward layer that the model has .

**But in what this layer helps us to train  efficiently our model with less ressources ?**
(Video link : https://www.youtube.com/watch?v=HcVtpLAGMXo&t=60s)

This schema explains everything !

![schemaAdapters](adapters.png)During the training we only train the adapter's weights , and the others weights we freeze them .
Let's take a quick example :

- Let's considere a matrix with the following size (we skip the batch dimension) :

$$ 
M_{input} = (seqLen , 1024)
$$

- Then we put this matrix to the first adapter :

$$ 
Output =M_{input} +\sigma(M_{input} * W_{Down}^T + b) * W_{up}^T
$$

With

$$
W_{Down}^T = (1024,64)
$$

$$ 
W_{up}^T = (64,1024)
$$
For this adapter we will train **99K parameters**

$$ 
2 *16 * 1024 * 64 * 2 + 1024 + 64 = 4229120 
$$

  

We have two adapter layers so finally we will train : **4.2M** parameters instead of **345M** for **Bert Large**

It is **1.21%** of the total number of parameter

# 2) Method 2 : LoRa

LoRa method does not add a new layer to the architecture but instead we add to every learnable matrix **(for example the matrices of the multi-head attention W_Q, W_K, W_V)** and then we add to them another matrix called $\Delta W$. This matrix contains the **trainable weights**.

## But what contains $\Delta W$ ?

  

$$
\Delta W = B A
$$
With:

- $B \in \mathbb{R}^{r \times d_{in}}$ : projects the input vector into a low-rank latent space of dimension $r$.

- $A \in \mathbb{R}^{d_{out} \times r}$ : projects back from the low-rank latent space to the original output dimension.

- $r$ is a small number (e.g., 16, 32, 64) much smaller than $d_{in}$ and $d_{out}$, which makes the fine-tuning **lightweight**.

### Dimensions


| Matrix | Shape |
|--------|-------|
| Original W | [d_out, d_in] |
| B | [r, d_in] |
| A | [d_out, r] |
| ΔW = BA | [d_out, d_in] |

### Intuition

- $B$ compresses the input features into a smaller latent space.

- $A$ expands it back to the original output space.

- Each element of $\Delta W$ is a **linear combination of $r$ latent components**, creating a low-rank correction applied to $W$ during the forward pass.
### Forward pass

  

For an input $x$:

  

$$

\text{output} = x W^T + x \Delta W^T = x W^T + x (A B)^T

$$

  

- $x W^T$ → original output (weights frozen)

- $x \Delta W^T$ → trainable correction applied to the output

- The output keeps the **same dimension** as before.

  

## Number of trainable parameters compared to BERT-large

  

- **BERT-large** total parameters: ~335M

- Typical LoRA setup (r = 16) applied to all attention matrices (`W_Q`, `W_K`, `W_V`, `W_O`) :

  

$$

\text{trainable params} \approx 4 \times (d_{model} \times r + r \times d_{model}) \times n_\text{layers}

$$

  

- For BERT-large:

- `d_model = 1024`

- `r = 16`

- `n_layers = 24`

- Approximate trainable parameters:

$$

4 \times (1024 \times 16 + 16 \times 1024) \times 24 = 4 \times 32768 \times 24 \approx 3\,145\,728

$$

  
  

So about **~3.1M trainable parameters**, which is **less than 1%** of the full 335M parameters.

  

- Huge memory and computation saving while still adapting the model effectively to new tasks.

# 3) Comparisons:

We can compare adapters and LoRA in terms of trainable parameters, efficiency, and memory footprint.

Below is a summary based on the official paper.

  

![image.png](comp.png)## Conclusion :

Use LoRa for fine-tuning tasks !!!

## Let's code now that we have the theoery !!!!


```python
from transformers import (AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig)
from huggingface_hub import login
from datasets import load_dataset
from trl import SFTTrainer , SFTConfig
from peft import get_peft_model,LoraConfig, prepare_model_for_kbit_training
import numpy as np
import torch

```


```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_name = "mamachang/medical-reasoning"
```


1) Let's import our model but let's quantized it before with NF4 method seen in the following post [[Deep dive into quantization]]
```python
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4"
)
```

Now we can import our model without crashing our laptop this time ;)

```python
model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config = bnb_config)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
```

# Let's create our LoRa configuration that we will use use to fine-tune our model

```python
# Configuration LoRA Optimisée pour TinyLlama Chat
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=64,
    lora_alpha=128, # Alpha = 2 *r
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj",
        "o_proj", 
        "gate_proj", 
        "up_proj",  
        "down_proj", 
    ],
    lora_dropout=0.05,
    bias="none",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
dataset = load_dataset(dataset_name ,split="train

```



The dataset used has some tokens , so let's add them to our model vocabulary to avoird the tokenizer chunking them .

**Exemple**:

We want the tokenizer to understand the entire token `*<answer>*` and not to split it into 3 tokens for example :

**["<an","swe","r>"]**


```python
tokens =[
    "</answer>'",
    "<answer",
    "<analysis>",
    "</analysis>"
]
num_tokens_added = tokenizer.add_tokens(tokens)
print(f"We added {num_tokens_added} tokens !")
```


**Important : We need to reshape our embeddings because the old one did not have 4 rows representing the embeddings of each token .**


```python
model.resize_token_embeddings(len(tokenizer))
```

If you execute the code you will notice the following output :`Embedding(32004, 2048)` what shows us that now we have the embedding of these new tokens !

# Let's prepare ou dataset

The output has the following structure: 


```python
Dataset({
    features: ['input', 'instruction', 'output'],
    num_rows: 3702
})  

```

I will preprocess it to obtain something like this :
`<|system|>Instruction</s>
`<|user|>Input </s>`
`<|assistant|> Output</s>`


```python
def prepare_ds(example):
    full_prompt = (
            f"<|system|> {example['instruction']}</s>"
            f"<|user|> {example['input']}</s>"
            f"<|assistant|> {example['output']}</s>" 
        )
    return {"text": full_prompt}
dataset = dataset.map(prepare_ds).remove_columns(["instruction",'input','output'])
```

Let's finally create ou trainer with the **TrainingArguments** that we want ! **(I choose the max_tokens = 775 because it is the 95th percentile of the lengths of our dataset )**


```python
sft_config = SFTConfig(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_length=775
)
trainer = SFTTrainer(
    model,
    train_dataset= dataset,
    args= sft_config
)
```

Let's train it and see what happens !!

```python
trainer.train()
```

Never forgot to save the weights to reuse our fine-tuned model 

```python
merged_output = "./merged_output"
new_model_merged= model.merge_and_unload()
new_model_merged.save_pretrained(merged_output)
tokenizer.save_pretrained(merged_output)
```


# Test of our fine-tuned model :

```python
from transformers import AutoModelForCausalLM, AutoTokenizer ,pipeline 
import torch
from pprint import pprint
import textwrap
from tqdm import tqdm
```

We first load the modeland the tokenize that we have trained , and then we create a `pipeline` that will allows us to generate a text by using some cool parameters (the *max_new_tokens* for example)

```python
final_model = "./merged-model"
model = AutoModelForCausalLM.from_pretrained(final_model)
tokenizer = AutoTokenizer.from_pretrained(final_model)
pipe = pipeline('text-generation', model=model,tokenizer= tokenizer)
```

Then we create a simple function where we the user can simply write an input to have a answer generated by our model . 

The **most important** point here that help me because at the beginning my model was answering very randomly is to insert  an *instruction* (the one why you fine-tuned your model for). The model will then have more context .

```python
len_wrap = 70
def generate(text):
    instruction = ("<|system|>"
                    "Please answer with one and only one of the option in the bracket" 
                    "That have been given by the user"
                    "And explain your choice."
                    "</s")
    tex = ("<|user|>" "Q : "+text+"</s>")
    tex = instruction + tex
    answer = pipe(tex, max_new_tokens=130, top_k=50, temperature=0.7, return_full_text=False)
    if not answer or answer[0]["generated_text"] =='':
        print("Bug detected please rerun ")
    l= textwrap.wrap(answer[0]['generated_text'].replace("'",""),len_wrap)
    final_ans = "\n".join(l)
    print(final_ans)    
```

Here is an example :

```python
text= ("I have a headache and I vomit a lot . I am 22 years old and I smoke a lot. What do you think I have")
generate(text)
```

And the <u>generated answer</u>  is:

*<|assistant|>  
A: Based on the given scenario, the patient is an adult male with a history of smoking and headaches. The smoking history is significant, and the headache is a common side effect of smoking. The patients lifestyle factors and medical history have been properly assessed and recorded. Therefore, the answer to the question is "a smoker with a history of frequent headaches and vomiting". Since there is no other information provided, it is concluded that this patient has experienced frequent headaches and vomiting as a result of smoking.*

An other one for the fun :

```python
text = ( "Patient is a 45-year-old male with a history of hypertension and obesity. "
        "He presents with swelling in both lower legs and shortness of breath. "
        "These symptoms have worsened over the past 2 days. Given these risk factors and symptoms, "
        "the most concerning diagnosis is one of the following: "
        " { A) Congestive heart failure B) Deep vein thrombosis C) Pulmonary embolism}."
        )
generate(text)
```

Here is what the model thinks  about this situation:

*<|assistant>  
Sure, based on the patients history, symptoms, and risk factors, the most concerning diagnosis is moderate-severity congestive heart failure. Congestive heart failure is a chronic condition characterized by increased workload and decreased contractility of the heart muscle. It can cause swelling in the legs and shortness of breath due to fluid and blood pooling in the legs. The patients symptoms have worsened over the past 2 days, which is consistent with congestive heart failure.*


# Evaluation :

The question that I asked my self after learning and coding all this stuff is :  
**How can I evaluate what I have done ?**

For this we will use the notion of `perplexity` which is one the different methods used to evaluate llms in general and not only in fine-tuning
# Perplexity

**Perplexity** is a measurement used in **Natural Language Processing (NLP)** and **probabilistic modeling** to evaluate the quality of a language model. It indicates how "uncertain" the model is when predicting the next unit (word, character, or token) in a sequence.
## Definition

For a language model predicting a sequence of words $ w_1, w_2, ..., w_N $, perplexity is defined as:


$$

\text{Perplexity} = PPL = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_1, ..., w_{i-1})}

$$

  
- $ P(w_i \mid w_1, ..., w_{i-1})$ : the probability the model assigns to word $w_i $ given the previous context.

- $N$ : the total number of words in the sequence.
## Interpretation

- A **low perplexity** means the model predicts the next words well; it is less "perplexed."

- A **high perplexity** means the model struggles to predict the sequence correctly.
## Intuition

Perplexity can be thought of as the **average branching factor**: how many choices the model considers equally likely at each step.

- Example: A perplexity of 50 means that, on average, the model is as uncertain as if it had to choose between 50 equally likely options for the next word.

The lowest the perplexity is , the better is the model !

# Test dataset

Here I asked to chatgpt to generate 40 examples of differents queries to compute the perplixity of the model and evaluate it.


```python
test_ds = [
    "Patient is a 60-year-old woman with a history of hypertension. "
    "She presents with chest pain that started 2 hours ago and is radiating to her left arm. "
    "The pain is described as pressure-like (8/10). Given these risk factors and "
    "symptoms, the most concerning diagnosis is one of the following: "
    " { A) Acute myocardial infarction B) Angina pectoris C) Gastroesophageal reflux disease (GERD)}. Choose the best answer. "
    ,
    "Patient is a 70-year-old male with a history of diabetes and obesity. "
    "He presents with a headache that has been persistent for the last week, accompanied by blurred vision. "
    "His blood pressure is elevated at 160/90 mmHg. Given these risk factors and symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Hypertensive crisis B) Cluster headache C) Retinal detachment}."
    ,
    "Patient is a 45-year-old female with a history of frequent alcohol use. "
    "She presents with a 3-day history of nausea and vomiting, along with right upper quadrant pain. "
    "The pain worsens after meals. Given these risk factors and symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Acute pancreatitis B) Gallbladder disease (cholelithiasis) C) Peptic ulcer disease}."
    ,
    "Patient is a 55-year-old male who recently returned from a trip to Southeast Asia. "
    "He presents with a fever, chills, and muscle aches. His symptoms started 5 days ago. "
    "Given these risk factors and symptoms, the most concerning diagnosis is one of the following: "
    " { A) Dengue fever B) Malaria C) Influenza}."
    ,
    "Patient is a 30-year-old woman who presents with a 2-week history of a sore throat, fever, and fatigue. "
    "She also has swollen lymph nodes in her neck. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Mononucleosis B) Strep throat C) Tonsillitis}."
    ,
    "Patient is a 65-year-old male with a history of smoking and chronic cough. "
    "He presents with shortness of breath and a productive cough with yellow sputum. "
    "These symptoms have worsened over the past 2 weeks. Given these risk factors and symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Chronic obstructive pulmonary disease (COPD) exacerbation B) Pneumonia C) Lung cancer}."
    ,
    "Patient is a 40-year-old woman who presents with a new, sharp, left-sided abdominal pain that started suddenly 4 hours ago. "
    "She also has nausea and vomiting. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Ectopic pregnancy B) Appendicitis C) Ovarian torsion}."
    ,
    "Patient is a 75-year-old male with a history of atrial fibrillation. "
    "He presents with sudden onset of left-sided weakness and difficulty speaking. "
    "His blood pressure is 180/100 mmHg. Given these risk factors and symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Ischemic stroke B) Transient ischemic attack (TIA) C) Bell's palsy}."
    ,
    "Patient is a 50-year-old woman with a history of thyroid disease. "
    "She presents with weight gain, fatigue, and cold intolerance. "
    "These symptoms have gradually developed over the last 6 months. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Hypothyroidism B) Depression C) Chronic fatigue syndrome}."
    ,
    "Patient is a 25-year-old male with a history of anxiety. "
    "He presents with a racing heart, chest tightness, and dizziness that started 30 minutes ago. "
    "Given these symptoms, the most concerning diagnosis is one of the following: "
    " { A) Panic attack B) Acute myocardial infarction C) Anxiety disorder}."
    ,
    "Patient is a 60-year-old woman who presents with a 4-day history of low back pain that radiates down her left leg. "
    "She also has numbness in her left foot. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Sciatica B) Spinal stenosis C) Herniated disc}."
    ,
    "Patient is a 45-year-old male with a history of hypertension and obesity. "
    "He presents with swelling in both lower legs and shortness of breath. "
    "These symptoms have worsened over the past 2 days. Given these risk factors and symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Congestive heart failure B) Deep vein thrombosis C) Pulmonary embolism}."
    ,
    "Patient is a 50-year-old woman with a history of asthma. "
    "She presents with wheezing and increased difficulty breathing, which began 2 hours ago. "
    "She is using her inhaler more frequently. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Asthma exacerbation B) Pneumothorax C) Acute bronchitis}."
    ,
    "Patient is a 65-year-old male with a history of chronic kidney disease. "
    "He presents with swelling in his face and hands, along with decreased urine output. "
    "His blood pressure is 150/95 mmHg. Given these symptoms, the most concerning diagnosis is one of the following: "
    " { A) Acute renal failure B) Nephrotic syndrome C) Hypertensive nephropathy}."
    ,
    "Patient is a 35-year-old woman who presents with a 2-week history of headaches, dizziness, and neck stiffness. "
    "She has also been feeling nauseous and feverish. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Meningitis B) Tension headache C) Cervical spondylosis}."
    ,
    "Patient is a 40-year-old male who presents with a 1-week history of a cough and fever. "
    "He is also experiencing night sweats and unexplained weight loss. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Tuberculosis B) Pneumonia C) Sinusitis}."
    ,
    "Patient is a 28-year-old female with a history of chronic headaches. "
    "She presents with a severe headache (9/10) that started suddenly, along with nausea and vomiting. "
    "She has a history of oral contraceptive use. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Subarachnoid hemorrhage B) Migraine C) Cluster headache}."
    ,
    "Patient is a 60-year-old male smoker who presents with a persistent cough and hemoptysis. "
    "He also reports weight loss and fatigue. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Lung cancer B) Tuberculosis C) Chronic obstructive pulmonary disease (COPD)}."
    ,
    "Patient is a 55-year-old male with a history of diabetes. "
    "He presents with a painful, swollen left foot. He reports a history of poorly controlled blood sugar and numbness in his foot. "
    "Given these symptoms, the most concerning diagnosis is one of the following: "
    " { A) Diabetic foot ulcer B) Gout C) Cellulitis}."
    ,
    "Patient is a 70-year-old female with a history of osteoarthritis. "
    "She presents with joint pain and stiffness, particularly in the knees, which worsens in the morning. "
    "Given these symptoms, the most concerning diagnosis is one of the following: "
    " { A) Osteoarthritis B) Rheumatoid arthritis C) Gout}."
    ,
    "Patient is a 55-year-old male with a 2-week history of progressive fatigue and weight loss. "
    "He also has a low-grade fever and night sweats. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Leukemia B) Hodgkin lymphoma C) Tuberculosis}."
    ,
    "Patient is a 40-year-old woman with a history of multiple pregnancies. "
    "She presents with a sudden onset of severe right lower quadrant pain. "
    "She also reports nausea and vomiting. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Ovarian cyst rupture B) Appendicitis C) Ectopic pregnancy}."
    ,
    "Patient is a 60-year-old female with a history of chronic pain management. "
    "She presents with increased pain in her lower back and bilateral leg weakness. "
    "She has difficulty walking and reports numbness in her feet. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Spinal cord compression B) Osteoarthritis C) Peripheral neuropathy}."
    ,
    "Patient is a 45-year-old male with a history of heavy alcohol use. "
    "He presents with jaundice, abdominal pain, and dark urine. "
    "His liver enzymes are elevated. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Hepatitis B C) Alcoholic cirrhosis D) Gallstones}."
    ,
    "Patient is a 50-year-old female who presents with progressive shortness of breath and fatigue. "
    "She has a history of hypertension and is now noticing swelling in her ankles. "
    "Given these symptoms, the most concerning diagnosis is one of the following: "
    " { A) Heart failure B) Pulmonary embolism C) Chronic obstructive pulmonary disease (COPD)}."
    ,
    "Patient is a 28-year-old male who presents with painful urination, fever, and chills. "
    "He reports frequent urination and lower abdominal pain. "
    "Given these symptoms, the most concerning diagnosis is one of the following: "
    " { A) Urinary tract infection (UTI) B) Pyelonephritis C) Bladder cancer}."
    ,
    "Patient is a 33-year-old woman who presents with a rash on her face and joints. "
    "She reports a butterfly-shaped rash across her cheeks and nose. "
    "She also has joint pain and swelling in her hands. Given these symptoms, "
    "the most concerning diagnosis is one of the following: "
    " { A) Systemic lupus erythematosus B) Rheumatoid arthritis C) Psoriatic arthritis}."
]
```

# 1) Tokenize every sentance of the test_ds

But before we need to determin the best length to padd the others sentances that have less tokens than this value.

For this I decided to take the as value the **95-th percentile** of the list containing the length of every tokenized sentance.


```python
import numpy as np
tes = tokenizer(test_ds)["input_ids"]
tab_len = [len(sub) for sub in tes]
tab_num = np.array(tab_len)
mean = round(np.mean(tab_num))
percentile_95 = round(np.percentile(tab_len,95))
print(f'The mean number of tokens per sentance is : {mean} tokens')
print(f'The 95 th percentile of the lengths is : {percentile_95}')
MAX_TOKENS = int(percentile_95)

def tokenize(ds):
    """ 
    This function tokenize the input dataset and returns a tensor by using as max_length: MAX_TOKENS 
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(ds,return_tensors="pt",max_length= MAX_TOKENS, truncation= True , padding= 'max_length')
tokenized_ds = tokenize(test_ds)
```


# 2 ) How to calculate the perplexity now ?

The formula is :
$$
\text{Perplexity} = PPL = e^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_1, ..., w_{i-1})}
$$

- $ P(w_i \mid w_1, ..., w_{i-1})$ : the probability the model assigns to word $w_i $ given the previous context.

- $N$ : the total number of words in the sequence.
# But there is one problem ! 

As you may know , LLMs have some limits, and one these limits is : **the window context**
<u>**Quick explanation:**</u>

A LLM uses the previous tokens that he generated to actually generate new tokens : that is the context . But this context is not unlimited , he can only **"attend to"** a limited number of token (this number depends on the model used).


In our case as you have seen we sum the $log_2 P(w_i \mid w_1, ..., w_{i-1})$ over the number of words . But the issue is that if we want to compute this for a long text that is longer than the size of the context window we will be forced to chunk the text and then at every new chunk , **the model will completly lose the previous context** and then the model will typically yield a higher (worse) PPL because the model will have less context at most of the prediction steps.
# So what is the solution ?
The solution is to use a sliding window we will repeatedly slide the context window so that the model has more context when making each prediction.


(Here you can find a lovely animation https://huggingface.co/docs/transformers/perplexity)

Here is the code :

```python
torch.cuda.empty_cache()
test_data = tokenized_ds.input_ids
device = "cuda" if torch.cuda.is_available() else "cpu"
stride = 70
w_context = 100  
seq_len = test_data.size(1)
nll_sum  = 0.0
n_tokens = 0
previous_end = 0
for start in tqdm(range(0,seq_len, stride)):
    end = min(seq_len,start+w_context)
    input = test_data[: , start:end].to(device)
    overlap = end - previous_end
    tgt = input.clone()
    #We mask the overlaped tokens to not compute their loss
    tgt[:,:-overlap] = -100
    with torch.no_grad():
        outputs= model(input, labels = tgt)
        neg_ll = outputs.loss
    num_valid_tokens = (tgt != -100).sum().item()
    batch_size = test_data.size(0)
    num_loss_tokens = num_valid_tokens - batch_size  
    nll_sum += neg_ll * num_loss_tokens
    n_tokens += num_loss_tokens
    previous_end = end
    if end == seq_len:
        break
avg_nll = nll_sum/n_tokens
ppl = torch.exp(avg_nll)
```

We finally obtain  (on the test dataset):
$$
PPL = 29.2 
$$
To be honest , at first I tought that this result was bad (in fact it is) , but the think is that we didn't use enough data to train the model and I am sure that I didn't try to optimize the pipeline .


# In reality we don't need to do this because our context window is sufficient (sorry)


```python
def perplexity(ds):
    input_ids = ds.input_ids
    nll_sum = 0.0
    n_tokens = 0
    for t in input_ids:
        t = t.unsqueeze(0)
        t = t.to(device)
        labels = t.clone().unsqueeze(0)
        with torch.no_grad():
            outputs = model(t, labels=labels)
            neg_ll = outputs.loss
        num_tokens = (labels != -100).sum().item()
        nll_sum += neg_ll * num_tokens
        n_tokens += num_tokens
    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    return ppl
result = perplexity(tokenized_ds)
print(f"The perplexity of our fine-tuned model is :{result.item()}")
```

The result is a little bit better : 
$$
PPL =  26.03
$$

This is the end of this post , I hope you enjoyed It and learned something new .   
Thank you for reading it  and as usual :

$$ 
Keep\;Grinding !
$$
