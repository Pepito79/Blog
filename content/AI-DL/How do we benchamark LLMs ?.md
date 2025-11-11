---
title: How do we benchamark LLMs ?
draft: false
tags:
  - AI
  - Benchmark
---

The last project that I did showed me that the most important thing is to know if your model generates good answers or not . And to do this we need to evaluate it  , so for that we do what you probably know as : **benchmarking**.

But In reality there are a lot of **benchmarks** , each benchmark evaluate the model on a specific aspect. 

The goal of this post will be to explain who the most famous and most used benchmarks are working and how to intreprete and compare them .

# 1) Measuring Massive MultiTask Language Understanding (MMLU)

This benchmark had 57 different categories , from *Sociology* to Abstract *algebra*. Every categorie is filled with different multiple choice question. The goal is to show where the model might excel and where it might be lacking .

For every question , four answers are proposed  and only one is correct.
Here is a quick example:  

![Quizzxample](mmlu1.png)
Then we compute the accurary for each discipline : Humanities , Social Science , STEM and Other.

If you pay attention to the picture above , you probably asked yourself : what the hell  does few -shot means ?

When we talk about few-shots , we means that before giving the question to the LLM we give him 1 to 10 demonstration examples with answers to the prompt before appending the question.
Without these  **few-shots** the model performs very poorly !


# 2) AGIEval   

This benchmark evaluates the abilities of LLMs  to tackle human-level tasks , it tests the model on  in the context of human-centric standardized exams, such as college entrance exams, law school admission tests, math competitions, and lawyer qualification tests 

There are two types of questions :
- Multiple-choice questions (MCQ)
- Fill in the blank questions
For MCQ we use accuracy as in the MMLU and for the fill in the blank questions we use F1 and Exact Math **(EM)** metrics.

**<u>EM metric</u>**
The model need to give the exact answer , THE EXACT ONE , this metric is very strict.
*Quick example:*   

Q: What is the  capital of France ?
Correct Answer : Paris 

If the model answers: "The capital is Paris" --> EM = 0
If the model answers : 'Paris' --> EM = 1

Here is another explanation of the few shot concepts (https://arxiv.org/abs/2304.06364)  

![agieEval](AGIEval.png)
# 3) ARC-Challenge (Ai2 Reasoning Challenge)

The ARC benchmark aims to test the scientific raisoning of a model ,  and more precisely the multi-step reasoning . This is the main difference between this benchmark and the others is the difficulty and the fact that the model needs to combine multiple concepts to answer.

Here is the distribution of the different reasoning type questions .  

![ARC](ARC.png)

That was It , short post to talk about these different methods .  

In the meantime :   

$$
 
Keep\;Grinding !

$$