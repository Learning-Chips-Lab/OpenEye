#!/bin/bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
F1=big18c_t2top.v
F2=big18c_a1mod.v
chk() { # $1=label  rest=args
  local L=$1; shift
  rm -rf m18g_$L
  verilator "$@" --top-module topT2 --dumpi-graph 6 -Mdir m18g_$L "$F1" "$F2" >m18g_$L.log 2>&1
  local RC=$?
  local F=$(ls m18g_$L/*inlinemod-decision.dot 2>/dev/null | head -1)
  if [ -z "$F" ]; then echo "$L: rc=$RC NO DOT: $(tail -2 m18g_$L.log | head -1)"; return; fi
  local R=$(python3 dump_inline_why.py "$F" a1 2>/dev/null | grep -c "NoInlineSoft: Public module" || true)
  echo "$L: a1-public=$R"
}
chk g1 --lint-only
chk g2 -cc
chk g3 -cc --lint-only
