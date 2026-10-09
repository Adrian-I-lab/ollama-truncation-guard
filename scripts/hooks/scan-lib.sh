#!/usr/bin/env bash
# Shared scanner for the pre-commit and commit-msg hooks.
# Reads stdin. Prints the CATEGORY of each hit, never the matched text.
# The private term list is read only through $PORTFOLIO_DENYLIST.

denylist_ok() {
  local f="${PORTFOLIO_DENYLIST:-}"
  if [ -z "$f" ]; then
    echo "BLOCKED: PORTFOLIO_DENYLIST is not set (failing closed)." >&2; return 1
  fi
  if [ ! -r "$f" ] || [ ! -s "$f" ]; then
    echo "BLOCKED: term list is missing, unreadable, or empty (failing closed)." >&2; return 1
  fi
  if grep -q '^[[:space:]]*$' "$f"; then
    echo "BLOCKED: term list has a blank line, which would match everything (failing closed)." >&2; return 1
  fi
}

# Category names and extended regexes. These are patterns, not secrets.
PATTERNS=(
  "private-key|BEGIN [A-Z ]*PRIVATE KEY"
  "github-token|(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
  "api-key|sk-[A-Za-z0-9_-]{20,}"
  "aws-key-id|AKIA[0-9A-Z]{16}"
  "slack-token|xox[abpors]-[A-Za-z0-9-]{10,}"
  "private-ipv4-10|(^|[^0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)"
  "private-ipv4-172|(^|[^0-9.])172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)"
  "private-ipv4-192|(^|[^0-9.])192\.168\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)"
  "cgnat-ipv4-100.64/10|(^|[^0-9.])100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}([^0-9]|$)"
  "home-path|/(home|Users)/[A-Za-z0-9_]"
)

# scan_text LABEL < text. Returns 1 if anything matched.
scan_text() {
  local label="$1" text hit=0 entry name re
  text="$(cat)"
  [ -z "$text" ] && return 0
  if printf '%s\n' "$text" | grep -q -F -i -f "$PORTFOLIO_DENYLIST"; then
    echo "BLOCKED: category=private-term in $label" >&2; hit=1
  fi
  for entry in "${PATTERNS[@]}"; do
    name="${entry%%|*}"; re="${entry#*|}"
    if printf '%s\n' "$text" | grep -q -E -e "$re"; then
      echo "BLOCKED: category=$name in $label" >&2; hit=1
    fi
  done
  return "$hit"
}
