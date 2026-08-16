# Morrow Field authentication integration

The portfolio installs `nexuss-auth` version 0.2.0 and creates a browser client with the public service URL and a registered project ID.

```bash
pnpm add nexuss-auth
```

Configure the client application with these public values through its deployment settings:

```text
VITE_NEXUSS_AUTH_URL=https://nexuss-auth.vercel.app
VITE_NEXUSS_AUTH_PROJECT_ID=<your registered project id>
```

The exact application address must be registered as the project homepage and the exact browser callback address must be registered as an allowed redirect URI. The portfolio uses its current browser address as the redirect URI, so the deployed URL must be registered exactly. Google and GitHub provider configuration remains on Nexuss-auth; no provider secret belongs in this portfolio.

The page calls `auth.getUser()` on load, uses `auth.signIn("google")` or `auth.signIn("github")` for access, and calls `auth.logout()` to end the session. If the project ID is missing, the page stays honest and displays a connection-required state instead of claiming that a visitor is signed in.
