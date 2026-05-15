

***Some typescript tools that you need to know***  

- ***Typescript compiler (tsc)***  : it performs type checking on the TypeScript code , compiles TypeScript syntax features to equivalent JavaScript and can generate some declaration files 
- **The Bundler , tsup** :  it reads your TypeScript files , transforms it into optimized JavaScript files (`.js` and `.d.ts`)  and **save them physically** in your **hard drive** (int the ./dist folder)
	- There is 3 main steps for the bundler :
		- 1)  **Packaging**: merge all the imports in one file in the ./dist folder
		- 2) **Transpilation**: transform .ts to .js files 
		- 3) **Minting and tree shaking**: delete spaces , comments and decrease variables name length and delete "dead code" (code that has not been used ) to save space
- **The Runner (tsx : typescript execute)**: it reads .ts files , converts them into .js but store them in the **RAM of your computer** and execute them immediatly with node . ***It does not create any file in your hard drive !***

# ***What is a build and what it does**

When we writte code , we write it in a language : typscript , go or python for example. But the thing is that your computer does not understand this , your computer only understand bits : 0 and 1.

The ***Build*** allows you to :
- **Compile** (translate) your code into 0 and 1 or into a simplest language (js for the web for example) 
- **In Turbo:** Usually runs with `--noEmit` to validate the code across the whole monorepo without actually generating files.
- **Assemble everything** , it groups every frames (files , images and extern librairies ) into one unique package.
- **Clean and optimize** :  delete the unecessary comments and decrease the files size to accelerate the execution of your app (**minting + tree shaking**)
- **Secure** your app by verifying that there is not any error in it

# **Main steps of a build:**

1) **Linting (verfication)**: we make sure that our code respect syntax rules and have the same style and also avert us if there is some unused variables  or some potential bugs .

2) **Compilation** : transform files from .ts to .js in our case

3) **Minification**: For the web we decrease the size of the files 

4) **Tests:** we verify if the core functions work correctly , if there is a bug the build process stops immediatly , it avoids us to push bugs to the production env.

5) **Packagaging** : one the code compiled and tested we packaged it in a format that can be deployed . A **.exe** for windows app s, **.apk** for android apps or **/dist** folder for a website or webapp.

In a NextJS app the configuration of the build process is in the **package.json** files where there is a **scripts** section.

# **Turborepo** 

It is an amazing tool for monorepo ,  if you have 4 packages in the repo and you only modified one button , a  "normal"  build will build everything , also the pacakges that have not been changed , with **Turborepo** we only build what have been changed . In consequence we go from  a 15 minutes build to a **30 second** build !

Here are the main **concepts** that I found intersting for this tool:

### 1) Dependency Graph (The `^` Power)

Turborepo maps out how your packages relate to each other. Using the `^` symbol (e.g., `^build`), it ensures tasks are executed in the **correct order**: it builds dependencies first before building the dependent project, preventing compilation errors.

**Quick example** : 
Let's imagine we have an **App A** that depends on a **Lib B**  and we run **turbo build**, what happens ?

1) Tu[]()rbo go and looks at the **turbo.json** file and finds out the following command for the App A: *dependsOn: ["^build"]* 
2)  He understands that App A depends on some other librairies , so he stops the Build of the App A  and finishes the build of the **Lib B**
3) Once the build of the lib B finished he returns and finishes the build of the App A what prevents our app from compilation errors

### 2) Intelligent Caching

It never performs the same task twice. By hashing your source code and environment variables, Turbo remembers the result. If nothing has changed, it **instantly restores** the output from the local or remote cache instead of rerunning the process.

### 3) Parallel Execution

Unlike traditional tools that run tasks one after another, Turborepo executes every independent task **simultaneously**. It maximizes your CPU usage to finish the entire pipeline (build, lint, test) in the shortest time possible

### 4) Precise Input/Output Control

You can define exactly which files trigger a rebuild. For example, you can tell Turbo to **ignore test files** during a production build. This keeps the cache "warm" and avoids unnecessary work for changes that don't affect the final product.
### 5) Remote Caching (Team Speed)

Turbo can share its cache across a whole team or CI/CD pipeline. If a teammate has already built a specific version of a library, your machine will simply **download the result** instead of compiling it yourself.



* **How does pnpm workspace works ?** 
	pnpm-workspace.yaml is at the root of a project , this files tells to **npm** where to find internal packages.

	You have a **packages** section in this file where you put all the **repos** where pnpm will goes and find all the packages thanks to their **name** defined in the **package.json** file.

	*<u>Here is a simple example to understand how it works:</u>*  

	Here is the **pnpm-workspace.yml**:
	
	```ts
		packages: 
			-'apps/*' 
			-'packages/*' 
			- 'modules'
	```
	
	Then let's define a **package.json** for a **UI folder** which is in the packages folder:
	
	```ts
		{
	  "name": "@pepito/ui",  <-- This is the name that pnpm-workspace uses
	  "version": "1.0.0",
	  "main": "./dist/index.js",
	  "types": "./dist/index.d.ts",
	  "scripts": {
	    "build": "tsup index.ts"
	  }
	}
	```
	
	And now let's imagine I need this package   in the my webapp which is in : **apps/webApp**.  
	So in the dependencies in the **package.json** we will use the name that we  gave to our package and **tells pnpm to not go and search for it in internet** but to go directly and ask the workspace file where to find it locally !
	
```javascript
{
	  "name": "web-app",
	  "version": "1.0.0",
	  "dependencies": {
	    "react": "^18.0.0",
	    "@pepito/ui": "workspace:*"
	  },
	  "scripts": {
	    "dev": "next dev",
	    "build": "next build"
	  }
	}
```

**So here is what happen when your run `pnpm install` at the root:**

1) **pnpm reads the `.yaml` file** to identify the workspace boundaries.
2) **It scans every folder** (e.g., `packages/ui`, `apps/webApp`).
3) **It reads the `"name"`** inside each `package.json`.
4) **It maps your project**, creating an internal index of all local packages.
5) **It links them**: If `apps/web` requires `@pepito/ui`, pnpm checks its map, sees that 
   `@pepito/ui` exists in `packages/ui`, and creates a local link between them