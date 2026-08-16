

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

- [ ] Audit the latest Morrow Field demo and npm/PyPI versions.
- [ ] Copy the latest demo into `DEMO/` and push it to the selected GitHub repository.
- [ ] Publish only if npm or PyPI is missing required latest changes.
- [ ] Confirm the external Vercel project and provide the accessible URL or exact remaining handoff.
