"""GitHub integration for PredGate.

Posts or updates PR comments, sets commit status checks,
and uploads HTML report as a CI artifact.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

COMMENT_MARKER = "<!-- predgate-verdict -->"


def post_or_update_pr_comment(
    repo: str,
    pr_number: int,
    comment_body: str,
    token: str | None = None,
) -> bool:
    """Post or update an idempotent PR comment.

    Searches for an existing PredGate comment (identified by a marker)
    and updates it, or creates a new one.

    Args:
        repo: GitHub repo in 'owner/repo' format.
        pr_number: PR number.
        comment_body: Markdown content for the comment.
        token: GitHub token (defaults to GITHUB_TOKEN env var).

    Returns:
        True if the comment was posted/updated successfully.
    """
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.warning("No GITHUB_TOKEN found; skipping PR comment")
        return False

    try:
        from github import Github

        g = Github(token)
        repo_obj = g.get_repo(repo)
        pr = repo_obj.get_pull(pr_number)

        # Add the marker to the comment
        full_body = f"{COMMENT_MARKER}\n{comment_body}"

        # Look for an existing PredGate comment
        existing_comment = None
        for comment in pr.get_issue_comments():
            if COMMENT_MARKER in comment.body:
                existing_comment = comment
                break

        if existing_comment:
            existing_comment.edit(full_body)
            logger.info(f"Updated existing PR comment on {repo}#{pr_number}")
        else:
            pr.create_issue_comment(full_body)
            logger.info(f"Created new PR comment on {repo}#{pr_number}")

        return True

    except ImportError:
        logger.error("PyGithub not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to post PR comment: {e}")
        return False


def set_commit_status(
    repo: str,
    sha: str,
    state: str,
    description: str = "",
    target_url: str = "",
    context: str = "predgate",
    token: str | None = None,
) -> bool:
    """Set a commit status check on GitHub.

    Args:
        repo: GitHub repo in 'owner/repo' format.
        sha: Commit SHA to set status on.
        state: One of 'pending', 'success', 'failure', 'error'.
        description: Short description of the status.
        target_url: URL to link to from the status.
        context: Status check name.
        token: GitHub token.

    Returns:
        True if the status was set successfully.
    """
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.warning("No GITHUB_TOKEN found; skipping commit status")
        return False

    try:
        from github import Github

        g = Github(token)
        repo_obj = g.get_repo(repo)
        commit = repo_obj.get_commit(sha)

        commit.create_status(
            state=state,
            description=description[:140],
            target_url=target_url,
            context=context,
        )

        logger.info(f"Set commit status '{state}' on {repo}@{sha}")
        return True

    except ImportError:
        logger.error("PyGithub not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to set commit status: {e}")
        return False


def verdict_to_github_state(verdict_status: str) -> str:
    """Convert a PredGate verdict status to a GitHub commit status state.

    Args:
        verdict_status: PredGate verdict (pass, warn, fail).

    Returns:
        GitHub commit status state.
    """
    mapping = {
        "pass": "success",
        "warn": "success",  # WARN is reviewable, not blocking by default
        "fail": "failure",
    }
    return mapping.get(verdict_status, "error")
