#!/usr/bin/env bash
# Back-compat dry-run wrapper.
exec "$(dirname "$0")/dryrun_videomae_official_ssv2_ft.sh" "$@"
