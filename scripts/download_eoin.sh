#!/usr/bin/env bash
set -euo pipefail

dataset='eoinamoore/historical-nba-data-and-player-box-scores'
version='515'
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="${root_dir}/data/raw/eoinamoore/v${version}"
manifest_dir="${root_dir}/data/profile"
mkdir -p "${output_dir}"

include_identity=false
if [[ "${1:-}" == "--include-identity" ]]; then
  include_identity=true
elif [[ -n "${1:-}" ]]; then
  printf 'Usage: %s [--include-identity]\n' "$0" >&2
  exit 2
fi

files=(Games.csv PlayerStatistics.csv TeamStatistics.csv)
sizes=(11208824 389667759 36020027)
hashes=(
  8b21e7e0ebfe25f1de48c18ec701e08a47974d32be006be044b85b2661dfff29
  8e5a33676661c3612bf87f185abb745254b09ab66cfee5aeccb5ae13d538c652
  37df1d8096252a9a38a1162872fdf85a2e413ea978c8b1d70edd3bd7f4399be9
)

for index in "${!files[@]}"; do
  file="${files[$index]}"
  expected="${sizes[$index]}"
  expected_hash="${hashes[$index]}"
  target="${output_dir}/${file}"
  if [[ -f "${target}" ]]; then
    actual_hash="$(sha256sum "${target}")"
    if [[ "$(stat -c %s "${target}")" -ne "${expected}" ]] || [[ "${actual_hash%% *}" != "${expected_hash}" ]]; then
      printf 'Existing file failed integrity check: %s\n' "${target}" >&2
      exit 1
    fi
    printf 'Verified existing file: %s\n' "${file}"
  else
    printf 'Downloading %s from %s, version %s\n' "${file}" "${dataset}" "${version}"
    curl --location --fail --show-error --retry 5 --retry-delay 3 \
      --continue-at - \
      --output "${target}.part" \
      "https://www.kaggle.com/api/v1/datasets/download/${dataset}/${file}?datasetVersionNumber=${version}"

    actual="$(stat -c %s "${target}.part")"
    actual_hash="$(sha256sum "${target}.part")"
    if [[ "${actual}" -ne "${expected}" ]] || [[ "${actual_hash%% *}" != "${expected_hash}" ]]; then
      printf 'Integrity check failed for %s: expected %s bytes and SHA-256 %s\n' "${file}" "${expected}" "${expected_hash}" >&2
      exit 1
    fi
    mv "${target}.part" "${target}"
  fi
done

printf 'SHA-256 checksums for dataset %s version %s:\n' "${dataset}" "${version}"
for file in "${files[@]}"; do
  sha256sum "${output_dir}/${file}"
done

if [[ "${include_identity}" == true ]]; then
  mkdir -p "${manifest_dir}"
  identity_manifest="${manifest_dir}/eoin-v515-identity-files.tsv"
  printf 'file\tsize_bytes\tsha256\n' > "${identity_manifest}"
  for file in Players.csv TeamHistories.csv; do
    target="${output_dir}/${file}"
    if [[ ! -f "${target}" ]]; then
      printf 'Downloading identity file %s from %s, version %s\n' "${file}" "${dataset}" "${version}"
      curl --location --fail --show-error --retry 5 --retry-delay 3 \
        --continue-at - \
        --output "${target}.part" \
        "https://www.kaggle.com/api/v1/datasets/download/${dataset}/${file}?datasetVersionNumber=${version}"
      mv "${target}.part" "${target}"
    else
      printf 'Using existing identity file: %s\n' "${file}"
    fi
    size="$(stat -c %s "${target}")"
    hash="$(sha256sum "${target}")"
    printf '%s\t%s\t%s\n' "${file}" "${size}" "${hash%% *}" >> "${identity_manifest}"
  done
  printf 'Identity file manifest: %s\n' "${identity_manifest}"
fi
