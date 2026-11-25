# Guide for AI agents

This repository collects the code behind Luxonis showcase projects: builds made for demos, blog posts and videos to show what OAK devices can do. The code is **unmaintained** and kept **for reference only**. It is not a library and has no shared code, build or CI. Each top-level folder is one self-contained project.

Small, minimal examples of individual OAK features belong in [oak-examples](https://github.com/luxonis/oak-examples), not here.

## Adding a new project

1. **One folder per project** at the repository root, named in kebab-case after the showcase (for example `underwater-360-camera`), not after the original repo name.
2. **Copy source only.** Bring over the files the original repo tracks, plus relevant uncommitted work. Do not copy recordings, datasets, downloaded videos, built `.oakapp` packages, virtual environments, `__pycache__`, build output (for example `.pio/`), tool binaries, or AI chat transcripts. Never copy credentials, tokens, `.env` files or tool config folders such as `.oakctl-data/`. Check for secrets before adding anything.
3. **Give the project its own `.gitignore`** covering its generated files (recordings, `*.oakapp`, `.venv`, `__pycache__`, and so on).
4. **Write the project `README.md` in this order:**
   1. `# <Showcase title>`. Use the title of the docs.luxonis.com page.
   2. A note that this is a showcase project built for a Luxonis demo, that the code is unmaintained and for reference only, with a link to [oak-examples](https://github.com/luxonis/oak-examples) for maintained, minimal examples.
   3. **Used for:** clearly state which showcase/demo the project was used for and what it does in that build.
   4. **Device:** the OAK device that was used (for example OAK 4 FFC), plus the camera modules, sensors and other hardware the code depends on.
   5. The YouTube video as a clickable thumbnail (GitHub does not render video embeds): `[![<video title>](https://img.youtube.com/vi/<id>/maxresdefault.jpg)](https://www.youtube.com/watch?v=<id>)`.
   6. A table with links to the **blog post** (discuss.luxonis.com/blog), the **YouTube video**, and the matching **docs.luxonis.com** page.
   7. The technical content: setup, how to run, and hardware notes. If the project already had a README, keep its content here and move its headings down one level so there is only one `#` heading.
5. **Add a row to the projects table in the root `README.md`** with the folder link, device, blog post, video and docs links.

### Finding the links

The docs pages for showcase projects live under `docs/overview/industry-examples/` in the `docs-content` repository. Each page embeds the YouTube video (`{% video youtube-id="..." %}`) and links the blog post in its closing `large-card`. The page's public URL is `https://docs.luxonis.com/overview/industry-examples/<category>/<page-name>`. Check that every link loads before adding it. If a link does not exist yet, leave a clear `TODO` instead of guessing a URL.

### What to leave alone

- Do not refactor, modernize or "fix" project code while importing it. The code should match what was used in the demo.
- Do not add shared tooling, packages or CI across projects.
- Keep the root `README.md` description as it is: this repo holds unmaintained showcase code for reference, and [oak-examples](https://github.com/luxonis/oak-examples) is the place for core, low-level, minimal-code examples.
