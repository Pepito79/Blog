

TanStack Router is a file based and type safe router for React. It focuses on keeping routing, navigation, and data fetching in one place while taking advantage of TypeScript.

In this article, I'll go through the concepts I found the most useful when starting with TanStack Router: file based routing, the router configuration, loaders, dynamic routes, and navigation.

---

# File Based Routing

The route structure is entirely based on the files and folders inside the `routes` directory.

To create a route like:

```
/a/b/c
```

you simply create the following structure:

```
routes/
└── a/
    └── b/
        └── c.tsx
```

The last file (`c.tsx`) contains the code executed when the user visits that URL.

If you also want a page for `/a/b`, create an `index.tsx` file inside folder `b`:

```
routes/
└── a/
    └── b/
        ├── index.tsx
        └── c.tsx
```

Everything starts with the `routes` folder. This is where every route of the application is defined.

For example, if you want a route like:

```
/place/$placeId
```

you create:

```
place.$placeId.tsx
```

If you only need a route without parameters, you simply create:

```
place.tsx
```

---

# Basic Route Structure

A route is created with `createFileRoute`.

```tsx
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/posts/$postId')({
  loader: ({ params }) => fetchPost(params.postId),
  component: PostComponent,
})

function PostComponent() {
  const { postId } = Route.useParams()

  return <div>Post ID: {postId}</div>
}
```

---

# The Loader

The `loader` is a function executed before the component is mounted.

Before routers like TanStack Router, we usually fetched data inside a `useEffect()` and displayed a loading spinner while waiting for the response.

With a loader, the data is fetched before the component is rendered. When the user reaches the page, the data is already available. The loading state is handled by the router instead of the page component.

The router is responsible for:

- waiting for the request,
- caching the result,
- keeping everything typed.

Inside the component, retrieving the data is straightforward:

```tsx
const data = Route.useLoaderData()
```

The returned value is already typed.

`useEffect()` is still useful, but mainly for UI behavior and interactivity rather than data fetching.

If your route contains path parameters, you can access them directly inside the loader through the `params` object.

```tsx
import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/posts/$postId')({
  loader: ({ params }) => fetchPost(params.postId),
  component: RouteComponent,
})

function RouteComponent() {
  const data = Route.useLoaderData()

  return (
    <div>
      {JSON.stringify(data)}
    </div>
  )
}

async function fetchPost(postId: string) {
  const res = await fetch(
    `https://jsonplaceholder.typicode.com/posts/${postId}`
  )

  return await res.json()
}
```

---

# The Root Route (`__root.tsx`)

`__root.tsx` is the root of the entire application.

Every route is rendered inside it.

Its main purpose is to act as the application's layout. This is usually where you place global components such as:

- a Toaster container,
- TanStack DevTools,
- global providers.

Later in the project, you may also want every route to have access to information about the current user, such as whether they are authenticated.

To do that, we create a context. This context becomes available to every route and can be used inside loaders or `beforeLoad` middlewares.

The application's home page is defined by `index.tsx`.

---

# Creating the Router

Once the routes exist, the next step is creating the router itself.

```tsx
export function getRouter() {
  const context = getContext()

  const router = createTanStackRouter({
    routeTree,
    context,
    defaultPreload: 'intent',
  })

  setupRouterSsrQueryIntegration({
    router,
    ...
  })

  return router
}

declare module '@tanstack/react-router' {
  interface Register {
    router: ReturnType<typeof getRouter>
  }
}
```

The router injects a shared context (for example `{ queryClient }`) into every route. The context type is declared in `__root.tsx` (`MyRouterContext`).

This is simply dependency injection.

The declaration block at the bottom is one of the most important parts.

It registers the generated `routeTree` inside TanStack Router. Once this is done, components such as `<Link />` and hooks like `useNavigate()` automatically know every route of your application.

As a result, you get:

- route autocompletion,
- compile-time validation,
- type safety.

---

# Navigation

There are two common ways to navigate between pages.

## Using `<Link>`

The simplest solution is the `<Link>` component.

```tsx
<Link to="/posts" />
```

Since the router knows every route, you get autocompletion while typing.

If the route does not exist, TypeScript reports the error before the application even runs.

## Using `useNavigate()`

Sometimes navigation depends on a condition.

A common example is redirecting the user after a successful login.

In this case, use the `useNavigate()` hook.

```tsx
export const Component = () => {
  const navigate = useNavigate()

  const handleClick = () => {
    navigate({
      to: '/login',
    })
  }
}
```

---

# Dynamic Routes

Suppose every post has its own URL:

```
/posts/$postId
```

Instead of creating one page per post, we create a dynamic route.

In TanStack Router, dynamic segments are prefixed with `$`.

Example:

```
routes/
├── posts/
│   └── index.tsx
└── posts.$postId.tsx
```

When creating a link, the parameter is passed through the `params` object.

```tsx
<Link
  to="/posts/$postId"
  params={{ postId: String(post.id) }}
/>
```

The object keys must match the parameter names defined in the route.

---

# Reading Route Parameters

There are two ways to use `useParams()`. This is something that is not always explained in tutorials.

## `Route.useParams()`

When you're inside the route file, use `Route.useParams()`.

```tsx
export const Route = createFileRoute('/posts/$postId')({
  component: RouteComponent,
})

function RouteComponent() {
  const { postId } = Route.useParams()

  return <div>You are on page {postId}</div>
}
```

This is a method attached to the `Route` object, not a hook.

It is fully type safe.

TypeScript already knows that `Route.useParams()` returns an object containing a `postId` string.

For example:

```tsx
const { } = Route.useParams()
```

VS Code immediately suggests `postId`.

If you try to access a parameter that does not exist, such as `userId`, TypeScript reports the error immediately.

This is one of the strengths of TanStack Router: everything is typed.

---

## `useParams()` Hook

Now suppose your component is defined somewhere else in the project instead of the route file.

Importing the `Route` object would create a circular dependency, since the route already imports the component.

The solution is to use the `useParams()` hook directly.

```tsx
import { useParams } from '@tanstack/react-router'

function MyChildComponent() {
  const params = useParams()
}
```

By default, this is not type safe because TypeScript does not know which route the component belongs to.

The solution is to specify the route explicitly.

```tsx
import { useParams } from '@tanstack/react-router'

function MyChildComponent() {
  const { postId } = useParams({
    from: '/posts/$postId',
  })
}
```

Once the `from` property is provided, the hook becomes as type safe as `Route.useParams()`.

---

# Conclusion

TanStack Router brings routing, navigation, and data fetching together while keeping everything strongly typed.

The file based routing system makes the project structure easy to understand, loaders simplify data fetching, and the router handles caching and loading before components are rendered. Features like typed navigation and typed route parameters also reduce many common mistakes during development.

If you're already using TypeScript in your React applications, TanStack Router fits naturally into the workflow and provides a routing experience where most errors are caught before runtime.