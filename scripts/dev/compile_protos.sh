#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
proto_root="${repo_root}/proto"
proto_file="${proto_root}/supervisor/v1/supervisor.proto"

# Single shared output (server + worker both import from here, per #111).
out_dir="${repo_root}/src/shared/grpc"

mkdir -p "${out_dir}"

PROTOLETARIAT_VERSION="3.3.10"

descriptor_set="$(mktemp)"
trap 'rm -f "${descriptor_set}"' EXIT

python -m grpc_tools.protoc \
  -I "${proto_root}" \
  --python_out="${out_dir}" \
  --pyi_out="${out_dir}" \
  --grpc_python_out="${out_dir}" \
  --descriptor_set_out="${descriptor_set}" \
  --include_imports \
  "${proto_file}"

# protoletariat rewrites protoc's absolute imports to relative ones. It caps
# protobuf below 6, incompatible with the GPU runtime's 6.30+, so it runs as an
# ephemeral uvx tool fed the descriptor set above rather than re-invoking protoc.
uvx --from "protoletariat==${PROTOLETARIAT_VERSION}" protol \
  --python-out "${out_dir}" \
  --in-place \
  --create-package \
  --exclude-google-imports \
  raw "${descriptor_set}"
