#!/usr/bin/env bash
set -euo pipefail

# Resumable + hash-verified download of a pinned vLLM 0.19.1 Linux wheel.
# This script does not install software or alter system environments.
# Run on Tang. The wheel remains ignored under results/vendor/.
cd "$(dirname "$0")/.."
mkdir -p results/vendor
name="vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl"
wheel="results/vendor/$name"
expected="71a87f46cafab4489c69a5c5c83b870d0235e5694d8222303d460576293dc719"
url="https://files.pythonhosted.org/packages/78/20/f41216b79c87372a9d03175f36fa1411ee61059ce8c557d2691722ea4aae/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl"

for attempt in 1 2 3 4 5 6; do
  echo "Attempt $attempt (existing bytes: $(stat -c %s "$wheel" 2>/dev/null || echo 0))"
  if curl --fail --location --continue-at - --connect-timeout 30 \
          --max-time 900 --retry 1 --retry-delay 5 \
          --speed-time 120 --speed-limit 1024 \
          --output "$wheel" "$url"; then
    actual="$(sha256sum "$wheel" | awk '{print $1}')"
    if [[ "$actual" == "$expected" ]]; then
      echo "SHA256_OK: $actual"
      exit 0
    fi
    echo "BLOCKED: wheel checksum differs from frozen PyPI digest" >&2
    exit 4
  fi
done

echo "BLOCKED: download incomplete. Keep partial ignored file for a future resumable retry." >&2
exit 2
