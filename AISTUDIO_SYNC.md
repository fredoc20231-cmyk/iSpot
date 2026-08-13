# Keeping the AI Studio copy in sync with GitHub

**Rule for now: GitHub is the source of truth. The Google AI Studio copy is a
read-only mirror of `main`. Do NOT edit code inside AI Studio until told
otherwise** — local edits in AI Studio diverge from the repo and interfere with
the fixes being merged here (they're the reason the app kept looking like an
"old build").

## Why

Google AI Studio (aistudio.google.com) is a Gemini build/playground
environment, not a git working tree. It has **no automatic "pull from GitHub"**
— nothing in it watches this repo and updates itself. So "syncing" is a manual
action you take: **re-import the app from the repo.** That overwrites the AI
Studio copy with whatever is on `main` — i.e. a hard mirror.

## How to sync (do this after any PR is merged)

1. In AI Studio, open / re-connect the app **from the GitHub repo
   `fredoc20231-cmyk/iSpot`, branch `main`.**
2. Re-importing replaces the AI Studio copy with the repo contents. That IS the
   update — there is nothing else to click.
3. Confirm you're on the latest commit of `main` before relying on the app.

## The two things that keep it from drifting

- **Never edit code in AI Studio** (for now). Any change made there is a local
  divergence that the next re-import will discard — and in the meantime it makes
  the copy behave differently from the repo.
- **Never let AI Studio save/push its copy back to GitHub.** That would push the
  divergent copy over the real source. Only changes reviewed and merged through
  this repo's pull requests should reach `main`.

## Important limitation

AI Studio Build apps are essentially the **frontend** only. iSpot is a
**FastAPI backend + frontend**. If AI Studio is hosting only `ispot/frontend/`,
much of the current behavior (viewer barcode alignment, demos, QC, clustering)
lives in the **Python backend**, which AI Studio does not run — so the app there
will look incomplete or stale no matter how carefully you sync. For the real,
current app, run the backend (the slim Docker image, or a host that clones
`main`), not AI Studio alone.

## When you want AI Studio editing back on

Tell me, and I'll lift the "don't edit" rule and we'll agree on a flow (e.g. AI
Studio edits → export to a branch → PR → merge) so edits and repo fixes stop
colliding.
