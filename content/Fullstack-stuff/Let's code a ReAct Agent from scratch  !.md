
Everyone is nowadays using Claude , OpenClaw and other tools to code or do other tasks , but have you ever asked yourself how it actually works ?

This quick post will try to explain the core concepts of agents and then we will implement a ReAct agent from scratch  in python , no Langchain  no CrewAI nothing , just you and your computer . So let's begin !

**I) Definition :**

When you use an LLM as Gemini or Grok you come with a question and then ask for answer , but in general you are not convinced with the first answer  , so you keep talking with your LLM until you find out what you want .  This simple loop is what defines a **naive agent** (this is the basic definition of it)

Okay but in reality modern agents have access to **tools** but how could they know when to use them ? They need to think of which tool to call in each situation , but for this they need to create a plan , init ? 

Now I can introduce the notion of **ReAct agent (ReAct for Reasoning /Action and not for the framework )** , it's simply an llm that has some tools and that observe his environnement, think about what he has to do ,choose a tool , use it and  so on : it's a loop ! **(from now then when you see the word agent thinks of the ReAct agent)**. 

Here is a quick example to let you understand what I mean: 

Imagine you want to organize a football match with your friends this night , what will you do ?

You will inconsciously create a plan in your head  (you normally think that you will need to ):

**1) The plan(Reasonning) :**
- Verify if your friends are available
	- Call friend 1
	- Call friend 2 (and so on until you gather the needed number)
- Check if you have a ball 
- Check if the pitch is available

Then when you have defined your plan you execute it :

**2) Execution (Acting):**
- Call the friends (the tool used here is your phone )
- Go to your garden to verify if you have your ball
- Check on the internet if the pitch is available  (the tool used here is your computer or phone )

**3) Observation :**
But unfortunetaly nothing happens as you wanted to  :(  , the friend 1 is in holidays and you ball has been taken by your brother . That's sad but you have to deal with it and find a solution . You just go the the first step and modify your plan , execute this new plan , observe it and then see if you succeeded .

Voilà you have understood how an **agent** works ! 
Here is a quick schema of how you could imagine this  (https://www.youtube.com/watch?v=hKVhRA9kfeM ): 
![[Pasted image 20260403202242.png]]







