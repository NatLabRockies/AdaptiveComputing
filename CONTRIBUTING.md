# Contributing

Thank you for considering contributing to this project!

Before your contributions can be accepted, you will need to sign the Contributor License Agreement (CLA). Please review the CONTRIBUTOR_LICENSE_AGREEMENT.md for details.

## Sign Your Commits

This project uses the Developer Certificate of Origin (DCO) to manage contributions. Every commit must be signed off to indicate your agreement to the DCO. To do this, use the -s flag when making a commit:

```bash
git commit -s -m "Your commit message"

This will add a line like the following to your commit message:

Signed-off-by: Your Name <your_email@example.com>

## Fixing an Unsigned Commit

If you forget to sign a commit and open a pull request, you will fail the DCO check. To fix this:

1. Amend your most recent commit to add the sign-off:

```bash
git commit --amend --signoff

2. Force-push the amended commit to update the pull request:

```bash
git push --force-with-lease origin your-branch-name

## Handling Multiple Unsigned Commits

If you have multiple unsigned commits, you can use Git’s interactive rebase to add the sign-off to each commit:

1. Start an interactive rebase for the last n commits:

```bash
git rebase -i HEAD~n

Replace n with the number of commits you need to sign.

2. In the editor, change pick to edit for each commit you want to sign off.

3. For each commit, Git will pause and allow you to amend the commit with the sign-off:

```bash
git commit --amend --signoff

4. After signing off each commit, continue the rebase process:

```bash
git rebase --continue

5. Once all commits are signed, force-push the changes:

```bash
git push --force-with-lease origin your-branch-name

For more information on the DCO, please see Developer Certificate of Origin.

