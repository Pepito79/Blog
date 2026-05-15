
***1) The stack :***
It is a region of the memory (RAM) used for static memory allocation and execution control. It works with the Last-In First-Out (LIFO) principle 

The compiler manages the stack , when a function is called : memory is pushed onto the stack when the function returns , that memory is "popped" and then available for reuse .


We can see it a a tour of coins where you can only remove and add coins on the top you can't touch them from the bottom or the middle of the stack 

In the left of the illustration there are the adresses in the RAM and on the left what the contain
![[Pasted image 20260402154909.png]] 

**What is the role between the CPU and the RAM ?**

When we run C code , the CPU fetch the data from the stack brings it into its own internal registers do calculations and writes the result back to the ram.

But now we have a problem , the stack is no more static , when we add something to the stack it's size increases and we need to track this . For this we introduce the : **Stack pointer (SP)**  . It's a **registry in the CPU** that stores the memory address of the current **top** of the stack in RAM .

![[Pasted image 20260402155807.png]]


In reality in your C code you may have multiple functions , so how all these **functions information live together in the stack ?** 

Each function call occupies a **contiguous** block of memory on the stack , it's called a **stack frame** and to delimits the end of a **stack frame** there is an other pointer stored in the CPU : **the base pointer** .

![[Pasted image 20260402160432.png]]

Then the CPU must also track where which instruction is actually executed .

![[Pasted image 20260402162843.png]]

For example here when the CPU executes the instruction **mov edi, 3 (which means put the value 3 in the register edi)** there is a pointer called the **instruction pointer (IP)** which is the CPU register that holds the address .  

**Let's see a quick example that explains how Assembly functions work (Video link of the example : https://www.youtube.com/watch?v=u_-oQx_4jvo&t=170s)**

*Here is the C code :*

```c
int result = 0 ;
void main () {
	result = add (3,4);
}

int add(int a , int b ){
	int sum = a + b ;
	return sum ;
}
```


And here is it's equivalent in assembly

![[Pasted image 20260402165054.png]]

**Here is the workflow :**

1) The CPU put 3 in the register edi 
2) The CPU put 4 in the register esi 
3) It calls the function add

But here is the problem : when we call the function add it returns at the end an result but how do we keep track of where we were executing ?  When the add function is called the **IP register contains : 0x0009** and then it jumps to **0x0020** how after executing this function it will returns to **0x0013**? 

The thing is that the instruction **call add** will **push** in the stack the return address after that the the function get executed .


![[Pasted image 20260402170037.png]] 


After this it's the add function that "controls" the CPU , so it need to create it's own stack frame.
Here is what happens:
- We save the the base pointer of the main function on the stack : **0xFFFD**
- Now that the RBP is safe the add function sets **the RBP to the current stack pointer RSP**
Here is the new result of the stack 

![[Pasted image 20260402172109.png]]

Now eveything is safe we can move forward and exectue the code of the add function !

Then we need to give to extend the stack to allocate to it 32 bits which is 4 bytes:
We do this by executing:   
-  **sub rsp , 4 **
![[Pasted image 20260402172809.png]]

Then we move the fist variable value in a register and add to it the value of the second one . 
- **mov edi , eax  ( edi <-- eax = int a)**
- **add eax , esi (eax <-- eax + esi)**

After this we will copy the result of our sum (**eax** ) to the allocated memory cases/
- **mov [rbp - 4] , eax    (Put in the memory cases eax )**

Okay now we have executed our function so we can destroy it's memory allocation . We just need to moove the RSP by adding to it 4 bytes !
- **add rsp , 4**

Then we need to regive the orignal RBP  , but remember we have saved it in the top of the stack ! 
We just need to **POP** it and put it in the **RBP**:
- **pop rbp (rbp <-- top of the stack)**
![[Pasted image 20260402174045.png]]


Finally we need to return to where the code must jumps after finishing executing the call function :

![[Pasted image 20260402174235.png]]

And now we are done !  We have see how an assembly code works ! 

I hope it helps you in some way and always :
**Keep grinding !**