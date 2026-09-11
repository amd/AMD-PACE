#!/bin/sh
# AOCL-DLP 5.3 and oneDNN bundle incompatible Xbyak symbols. Isolate AOCL's
# copy while retaining its static JIT registrars.
set -e
LD="$1"
OBJCOPY="$2"
NM="$3"
AOCL_A="$4"
OUT="$5"
WORK="$6"
PARTIAL="$WORK/aocl_dlp_partial.o"
MAP="$WORK/aocl_dlp_xbyak_redef.txt"

"$LD" -r --whole-archive "$AOCL_A" --no-whole-archive -o "$PARTIAL"

"$NM" "$PARTIAL" | awk '{print $NF}' | grep "Xbyak" | sort -u \
    | awk '{print $1" __aocldlp_"$1}' > "$MAP"

if [ ! -s "$MAP" ]; then
    echo "isolate_aocl_xbyak: no Xbyak symbols found in $AOCL_A" >&2
    exit 1
fi

"$OBJCOPY" --redefine-syms="$MAP" "$PARTIAL" "$OUT"
