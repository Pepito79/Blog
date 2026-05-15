
# What is Tsdown ?

Tsdown is one of the best (if not the best) library bundler for `js` and `ts` librariries.  It's build on top of `RollDown` which is written in Rust.  While Rolldown is a powerful and general-purpose tool, `tsdown` takes it a step further by providing a **complete out-of-the-box solution** for library authors.

# Getting started with it :

1) Install the package with your package manager  (use pnpm !)

```javascript
	pnpm add -D tsdown
```

2) Write your config file it allows you to define and customize your build settings in a centralized and reusable way . The `entry` parameters is the first file that the tsdown will read and then follows all the `import` and `export` to build the dependancy graph . Every **direct** or **indirect** import from your **entry** file will be included in the **final bundle** , what is not will be ignored (**tree-shaking**)   
   
   Here is an example of a `tsdown.config.ts` file with one entry
```javascript
import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: 'src/index.ts',
})
```

3) By default, `tsdown` bundles your code into the `dist` directory located in the current working folder. 

4) You also have a `watch` mode that allows tsdown to automatically re-bundle your code whenever changes are detected in the specified files or directories. This is particularly useful during development process.

```javascript
tsdown --watch
```

5) Normally the `tree-shaking` mode is enabled by default : it allows you to remove the **dead code** present in your files . Furthermore we can also enable  `minification` which compresses your code to reduce its size and improve performance by removing unnecessary characters, such as whitespace, comments, and unused code. Here is how you can enable it:

```javascript
import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: 'src/index.ts',
  minify: true, // Activate the minification
})
```


6)  It also generates declaration files (`.d.ts`) , which are files that provides type definitions that allow consumers of your library to benefit from TypeScript's type checking . In fact `tsdown` makes it easy to generate and bundle declaration files for your library, ensuring a seamless developer experience for your users. You can also enable it in the `tsdown.config.ts` file:

```javascript
import { defineConfig } from 'tsdown'

export default defineConfig({
  dts: true,
})
```

7) And finally instead of running `tsdown` command evey time you want to bundle you package you can add it to the script section in your **package.json** :

```javascript
"scripts": {
        "dev": "tsx watch src/index.ts",
    },
```


# Conclusion 

That was a nice tool that I discovered whil trying to contribute to some open source projects , I hope that It will help someone !

Meanwhile : **Keep grinding !**