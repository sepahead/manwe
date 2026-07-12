# Releasing Manwe

The first alpha is a GitHub/source prerelease. Do not publish the Rust crates or
Python distribution to a registry: Cargo has `publish = false`, and Python keeps
the `Private :: Do Not Upload` guard.

## Alpha checklist

1. Confirm out-of-band that every camera credential exposed in earlier public Git
   history has been revoked or rotated. Source cleanup or a history rewrite does
   not replace rotation.
2. Require a clean `main` worktree and a green branch CI run. Verify the Rust,
   Python, benchmark, audit, package, secret, and platform jobs—not only one job.
3. Confirm the release metadata agrees:
   - Cargo and benchmark: `0.2.0-alpha.1`
   - Python: `0.2.0a1`
   - tag: `v0.2.0-alpha.1`
4. Replace “unreleased” in `CHANGELOG.md` with the release date, commit, and push;
   wait for branch CI again.
5. Create a signed annotated tag on that exact green commit and push it:

   ```bash
   git tag -s v0.2.0-alpha.1 -m "Manwe 0.2.0 alpha 1"
   git push origin v0.2.0-alpha.1
   ```

6. Wait for tag CI. Tag events scan the checked-out tree but intentionally do not
   traverse all historical blobs; credential rotation is therefore a separate,
   mandatory release decision rather than something tag CI can prove.
7. Create a GitHub prerelease from the verified tag. Use the changelog as the
   release notes and keep the documented limitations intact. The automatically
   generated source archives are the release artifacts for this alpha.

Do not describe the release as consumer-compatible, CUDA-validated, Windows-
supported, or production-ready until the corresponding documented gates pass.
