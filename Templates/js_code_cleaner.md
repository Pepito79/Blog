<%* const clip = await tp.system.clipboard();

const cleaned = clip
    .split('\n')
    .map(line => line.trimEnd()) // 1. Enlève les espaces inutiles à la fin de chaque ligne
    .join('\n')
    .replace(/\n{3,}/g, '\n\n') // 2. Évite d'avoir trop de lignes vides à la suite
    .trim();                    // 3. Enlève les espaces au tout début et à la fin du bloc

// 4. Insère le code dans un bloc Markdown JS
tR += "```javascript\n" + cleaned + "\n```";
%>