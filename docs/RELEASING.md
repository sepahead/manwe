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
5. Create a signed annotated tag on that exact green commit. Verify both the
   signature and the commit to which the annotated tag peels before pushing:

   ```bash
   tag=v0.2.0-alpha.1
   release_commit=$(git rev-parse HEAD)
   git tag -s "$tag" -m "Manwe 0.2.0 alpha 1"
   git tag -v "$tag"
   test "$(git rev-parse "${tag}^{commit}")" = "$release_commit"
   local_tag_object=$(git rev-parse "refs/tags/$tag")
   ```

6. Push the exact tag ref, then verify that the remote ref names the same tag
   object. Record both `release_commit` and `local_tag_object` in the release
   notes so reviewers can identify the authenticated Git objects without relying
   on a movable tag name:

   ```bash
   tag=v0.2.0-alpha.1
   release_commit=$(git rev-parse "${tag}^{commit}")
   local_tag_object=$(git rev-parse "refs/tags/$tag")
   git push origin "refs/tags/$tag"
   remote_tag_object=$(
     git ls-remote --refs origin "refs/tags/$tag" |
       awk 'NR == 1 { print $1 }'
   )
   test -n "$remote_tag_object"
   test "$remote_tag_object" = "$local_tag_object"
   printf 'release commit: %s\ntag object: %s\n' \
     "$release_commit" "$local_tag_object"
   ```

7. Wait for tag CI. Tag events scan the checked-out tree but intentionally do not
   traverse all historical blobs; credential rotation is therefore a separate,
   mandatory release decision rather than something tag CI can prove.
8. Create the GitHub prerelease as a draft from the verified tag, use the
   changelog as the release notes, include the recorded Git object IDs, and keep
   the documented limitations intact. If immutable releases are enabled for the
   repository or organization, publish the completed draft and verify that
   GitHub marks it immutable; `gh release verify "$tag"` can then check the
   published release record. If that protection is unavailable, state that the
   release remains administratively mutable and treat the signed tag object as
   the authentication root only after verifying it against the maintainer's
   trusted signing identity.
9. GitHub's automatically generated source ZIP and tarball are convenience
   snapshots, not byte-reproducible signed assets: GitHub may regenerate them
   with different compression, and `gh release verify-asset` cannot verify them.
   Consumers who need authenticity should fetch the Git tag, verify its
   signature, and confirm the recorded peeled commit. Do not publish archive
   checksums as stable release guarantees. A future release that ships binaries
   or a byte-stable archive must attach explicit assets plus checksums and
   provenance/attestations before the release is made immutable.

Do not describe the release as consumer-compatible, CUDA-validated, Windows-
supported, or production-ready until the corresponding documented gates pass.
