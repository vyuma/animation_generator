# Repository Guidelines

## Project Structure & Module Organization
Routes live inside `app/`: `app/page.tsx` is the default experience, `app/LP` and `app/test` house alternates, and `app/layout.tsx` wires shared `<Providers />`. UI modules sit in `app/components/`, while contexts/hooks live in `app/context/` and `app/hooks/`, and shared logic or types sit in `app/utils/`, `app/lib/` (animation API client), and `app/type/`. Static assets stay in `public/`, while root configs (`next.config.ts`, `tsconfig.json`, `eslint.config.mjs`, PostCSS/Tailwind files) tune the toolchain.

## Build, Test, and Development Commands
- `npm run dev` — starts the dev server on `http://localhost:3000` for `/` and `/test`.
- `npm run build` — creates the optimized production bundle.
- `npm run start` — serves the last build output and mirrors production routing.
- `npm run lint` — runs the Next + ESLint suite; use it before every commit.

## Coding Style & Naming Conventions
Write modern TypeScript with async/await and prefer server components unless interactivity demands a `'use client'` directive. Components, contexts, and hooks use PascalCase filenames (see `app/components/TextInputForm.tsx`), while utilities and functions use camelCase. Follow the repo’s two-space indentation, keep imports path-aware via the `@/app/...` alias, and rely on Tailwind 4 utilities plus `app/globals.css` tokens for styling.

## Testing Guidelines
Automated specs are not yet wired up, so linting plus manual QA are the current gatekeepers. When adding coverage, colocate React Testing Library tests as `Component.test.tsx` next to the source, mock fetches with MSW to keep `app/lib/animationApi.ts` deterministic, and target roughly 80 % line coverage for new modules. Always document manual test steps in the PR body.

## Commit & Pull Request Guidelines
Recent history (`git log --oneline`) shows concise, single-line subjects (`coomit`, `新しいUIを提供するための布石`), so keep commits imperative, under ~60 characters, and add a body when extra context is needed; bilingual summaries are fine if they stay clear. Each PR should link the relevant issue, describe the motivation, include before/after screenshots or a short recording for UI changes, and list manual test steps plus any `.env.local` updates. Request review only after `npm run lint` and `npm run build` succeed locally.

## Security & Configuration Tips
Sessions are stored in the browser (localStorage); there is no login. Set `API_URL` (default `http://localhost:8888`) to point at the backend. API calls flow through `app/api/*` route handlers, so keep backend URLs or tokens server-side and document any new proxy endpoint. When sharing logs or repro steps, scrub job IDs that may expose user data.
