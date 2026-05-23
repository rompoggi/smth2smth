#!/usr/bin/env bash
# Back-compat wrapper — espadon ep50 + mae50-ft-f16 defaults.
# See scripts/launch_videomae_official_ssv2_ft.sh for the generic launcher.
exec "$(dirname "$0")/launch_videomae_official_ssv2_ft.sh" "$@"
