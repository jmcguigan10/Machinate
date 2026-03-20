# Homebrew Release Notes

Machinator should be released from this repo, while the actual Homebrew tap lives in a separate repository such as `homebrew-tap`.

## Expected Repos

- app repo: `Machinator`
- tap repo: `homebrew-tap`

## Release Flow

1. Update the version in `pyproject.toml`.
2. Commit and tag a release in the `Machinator` repo.
3. Create a GitHub release for that tag.
4. Download the release tarball or compute its checksum.
5. Update the formula in the tap repo so that:
   - `url` points at the new tagged tarball
   - `sha256` matches the tarball
   - the `test do` block still passes
6. Push the tap repo changes.
7. Verify install from a clean machine or shell:

```bash
brew tap jmcguigan10/tap
brew reinstall machinator
macht --help
```

## Local Preflight Before Publishing

Before you push a release to GitHub, you can smoke-test the Homebrew packaging locally:

1. Build a source tarball from the app repo:

```bash
cd /Users/johnny/Projects/Machinator
./scripts/build_release_artifact.sh
./scripts/render_homebrew_formula.py --owner jmcguigan10
```

2. Render the tap formula from the current artifact checksum.
3. Register the local tap and install from it:

```bash
brew tap jmcguigan10/tap /Users/johnny/Projects/homebrew-tap
brew install jmcguigan10/tap/machinator
```

4. Verify:

```bash
$(brew --prefix)/bin/macht --help
```

The formula renderer writes a GitHub release asset URL in this shape:

```text
https://github.com/jmcguigan10/Machinator/releases/download/vX.Y.Z/machinator-X.Y.Z.tar.gz
```

That means the release asset uploaded to GitHub must be the same tarball produced by `build_release_artifact.sh`.

## Why The Tap Is Separate

Homebrew installs from formulae, not directly from your app repo's `pyproject.toml`.
That means the app repo and the tap repo move in lockstep, but they are still separate concerns:

- `Machinator` contains your source code and release tags
- the tap repo contains the formula Homebrew reads

## Prompt Dependency Strategy

Machinator now includes `questionary` in the default package install, so the brewed CLI gets the richer prompt UI automatically.
If prompt dependencies are unavailable in some runtime, Machinator still falls back to plain terminal prompts.
