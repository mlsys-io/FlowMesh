#!/bin/sh
# FlowMesh non-interactive entrypoint wrapper.
#
# Performs FlowMesh-specific container setup (input staging, output directory
# creation, finish-sentinel helper) and then exec's the user's process.
#
# Environment variables consumed:
#   FLOWMESH_STAGED_INPUT_SPECS  - newline-separated "mount_path\ttarget_path" pairs
#   FLOWMESH_CREATE_DIRS         - newline-separated list of directories to create
#   FLOWMESH_FINISH_SENTINEL     - path to the finish sentinel file
#   FLOWMESH_NO_FINISH_HELPER    - set on a read-only root: skip the helper
set -e

FLOWMESH_FINISH_SENTINEL="${FLOWMESH_FINISH_SENTINEL:-/tmp/.flowmesh_finish}"

# Optional FlowMesh mount helpers
if [ -n "${FLOWMESH_STAGED_INPUT_SPECS:-}" ]; then
    printf '%s\n' "$FLOWMESH_STAGED_INPUT_SPECS" | while IFS="$(printf '\t')" read -r mount_path target_path; do
        [ -n "$mount_path" ] || continue
        [ -n "$target_path" ] || continue
        if [ -e "$mount_path" ] && [ ! -d "$mount_path" ]; then
            echo "Refusing to replace existing non-directory path: $mount_path" >&2
            exit 1
        fi
        mkdir -p "$mount_path"
        cp -a "$target_path"/. "$mount_path"/
        chown -R root:root "$mount_path"
        chmod -R a-w "$mount_path"
    done
fi

if [ -n "${FLOWMESH_CREATE_DIRS:-}" ]; then
    printf '%s\n' "$FLOWMESH_CREATE_DIRS" | while IFS= read -r dir_path; do
        [ -n "$dir_path" ] || continue
        mkdir -p "$dir_path"
    done
fi

# In-session helper to finish the SSH task successfully. Not on a read-only
# root (a hardened session), which cannot hold it and does not honour it.
if [ -z "${FLOWMESH_NO_FINISH_HELPER:-}" ]; then
cat > /usr/local/bin/flowmesh-finish << EOF
#!/bin/sh
set -e
touch "$FLOWMESH_FINISH_SENTINEL"
echo "FlowMesh finish requested; the container will stop shortly."
EOF
chmod 755 /usr/local/bin/flowmesh-finish
fi

# Hand off to the user's process.
exec "$@"
