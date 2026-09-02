#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
proto_root="${repo_root}/proto"
proto_file="${proto_root}/supervisor/v1/supervisor.proto"

# Single shared output (server + worker both import from here, per #111).
out_dir="${repo_root}/src/shared/grpc"

mkdir -p "${out_dir}"

# protoletariat runs as an ephemeral uvx tool: it caps protobuf below 6, while
# the GPU inference runtime requires 6.30+, so the two cannot share a resolution.
# The protoc wrapper therefore names this project's interpreter explicitly —
# inside uvx, a bare `python` would resolve to the tool's own environment, which
# has no grpc_tools.
PROTOLETARIAT_VERSION="3.3.10"
project_python="$(python -c 'import sys; print(sys.executable)')"

# Create a temporary protoc wrapper script
temp_dir=$(mktemp -d)
protoc_wrapper="${temp_dir}/protoc"
cat > "${protoc_wrapper}" << EOF
#!/usr/bin/env bash
exec "${project_python}" -m grpc_tools.protoc "\$@"
EOF
chmod +x "${protoc_wrapper}"

# Add wrapper to PATH temporarily
export PATH="${temp_dir}:${PATH}"

# Cleanup function
cleanup() {
  rm -rf "${temp_dir}"
}
trap cleanup EXIT

python -m grpc_tools.protoc \
  -I "${proto_root}" \
  --python_out="${out_dir}" \
  --pyi_out="${out_dir}" \
  --grpc_python_out="${out_dir}" \
  "${proto_file}"

uvx --from "protoletariat==${PROTOLETARIAT_VERSION}" protol \
  --python-out "${out_dir}" \
  --in-place \
  --create-package \
  --exclude-google-imports \
  protoc -p "${proto_root}" "${proto_file}"
