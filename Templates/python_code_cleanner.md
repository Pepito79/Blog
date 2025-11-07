
```python
<%* 
const clip = await tp.system.clipboard();
const cleaned = clip.split('\n').filter(l => l.trim() !== '').join('\n');
tR += cleaned;
%>
```
