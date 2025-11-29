---
title: Get started with JS
draft: false
tags:
  - JS
---

# Introduction 






# Some basics commands 

1) To declare a variable that can be modified and has a limited scope (it means that it can only access in a block) we use : *const*
```js
const tab = []
```

2) To filter an array we can use *.filter* 
```js
const tab = [1,2,3]
const filterdTab = tab.filter(elem => elem> 2)
console.log(filteredTab) 
// It s output : [3]
````

3) To verify if an element belongs to an array we use *.includes*
````js

const fruits = ["Banana","Orange"]
const containsBanana = fruits.includes("Banana") 
// Return true
````

But if we have some custom object in our array we can not use *.includes* but *.some*
````js

const person = [
	{
		name: "saad",
		age: 18
	},
	{
		name:"John",
		age: 15
	}
]

const containsSaad = person.some( p => p.name === "saad") 

// Return true
````