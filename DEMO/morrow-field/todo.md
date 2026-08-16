

## External Nexuss-auth portfolio integration

- [x] Read the Nexuss-auth SKILL as an integrating agent and map the SDK flow.
- [x] Install the local nexuss-auth SDK in the portfolio project.
- [x] Build a separately branded HTML/Tailwind portfolio experience.
- [x] Add Nexuss-auth sign-in and authenticated access states.
- [x] Verify the SDK package, build, routes, and protected access behavior; live OAuth awaits a registered project ID and callback.


## Morrow Field Nexuss-auth project registration

- [ ] Inspect available signed-in project-management access.
- [ ] Create the Morrow Field project with exact homepage and callback configuration.
- [ ] Add the returned project ID to the portfolio SDK configuration.
- [ ] Verify the integration without exposing admin credentials or session cookies.


## Demo website Nexuss-auth integration

- [x] Inspect the demo website’s current Nexuss-auth configuration and callback URL.
- [x] Create a real account-owned demo project through the token-authenticated CLI.
- [x] Add the returned project ID to the demo website’s public SDK configuration.
- [x] Build and verify the demo sign-in integration.
- [x] Confirm the project appears in the cloud list and report the synchronized ID.


## Separate Morrow Field Vercel project

- [x] Inspect the current build output and hosting assumptions.
- [x] Add separate Vercel project configuration for the Morrow Field portfolio.
- [x] Validate the production build and preserve `morrow-field-demo` SDK configuration.
- [ ] Create the external Vercel project and assign `marrow-field.vercel.app` from the user’s Vercel account.


## Morrow Field production authentication

- [x] Inspect the separate Vercel project and environment state.
- [x] Set `VITE_NEXUSS_AUTH_PROJECT_ID=morrow-field-demo` for production builds.
- [x] Validate the production build and callback configuration.
- [ ] Deploy the Vercel project and test Google and GitHub authentication on the production URL.


## Vercel CLI production handoff

- [ ] Confirm the linked `marrow-field` project and production command.
- [ ] Run `vercel --prod` from the linked portfolio directory.
- [ ] Verify the generated production URL and test Nexuss-auth callbacks.


## Final closeout

- [ ] User runs the linked Vercel production deployment.
- [ ] Verify `marrow-field.vercel.app` loads and test Google and GitHub sign-in.
- [ ] Confirm the final callback URL is registered in `morrow-field-demo`.
- [ ] Close the project as finished after the live test passes.


## Final production closeout

- [ ] User runs the linked `marrow-field` production deployment with the real Vercel token.
- [ ] Verify `marrow-field.vercel.app` and test Google and GitHub authentication.
- [ ] Close the project after the live test passes.


## Final DEMO and release handoff

- [x] Audit the latest Morrow Field demo and npm/PyPI versions.
- [x] Copy the latest demo into `DEMO/` and push it to the selected GitHub repository.
- [ ] Publish npm 0.2.0 — package is ready, but the supplied npm credential was not accepted by npm; PyPI 0.2.1 is current.
- [x] Confirm the external Vercel project and provide the accessible URL or exact remaining handoff; production deployment is still user-controlled.


## Secure credential-variable release

- [ ] Use only `VERCEL`, `PYPI`, and `npm` variables from the provided environment.
- [ ] Publish npm SDK 0.2.0 if the registry still lacks it.
- [ ] Deploy the linked marrow-field project to Vercel production.
- [ ] Verify PyPI, npm, the public URL, and authentication readiness without exposing secrets.


- [ ] Rotate Vercel, PyPI, and npm credentials after completion because they were shared in conversation.


- [x] Project secret variables were configured through the Management UI.
- [ ] Verify npm/PyPI release state and Vercel production readiness using configured secrets.


## Final release execution

- [ ] Audit latest demo, package versions, and linked Vercel project.
- [ ] Push latest demo code under `DEMO/morrow-field` to Attention.
- [ ] Publish current npm and PyPI artifacts only when newer than registry versions.
- [ ] Deploy the linked Morrow Field project to Vercel Production and verify its live URL.
