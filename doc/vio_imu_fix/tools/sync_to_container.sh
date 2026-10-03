#!/usr/bin/env bash
# Copy the host dense_devel working tree (tracked + untracked, not ignored) into the container's
# dense_devel workspace (without doc/: notes, evaluation tools and results stay on the host). Never touches the
# simval checkout.
set -euo pipefail
HOST_REPO=/home/deos/s.jois/EXECO/SaDVIO-Dense/SaDVIO-EXECO
CONTAINER=sad_vio_dense
DEST=/root/SaDVIO-Dense/SaDVIO-EXECO-dense_devel

case "$DEST" in
  */SaDVIO-EXECO) echo "refusing to sync into the simval checkout" >&2; exit 1 ;;
esac

cd "$HOST_REPO"
branch=$(git branch --show-current)
[ "$branch" = dense_devel ] || { echo "host is on '$branch', expected dense_devel" >&2; exit 1; }

git ls-files --cached --others --exclude-standard -z \
  | grep -z -v -E '\.pdf$|^doc/' \
  | tar --null -T - -cf - \
  | docker exec -i "$CONTAINER" tar -xf - -C "$DEST"
echo "synced $(git ls-files --cached --others --exclude-standard | grep -v -E '\.pdf$|^doc/' | wc -l) files to $CONTAINER:$DEST"
