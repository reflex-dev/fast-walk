#!/usr/bin/env bash
# Writes the release tag's version (v1.2.3 -> 1.2.3) into Cargo.toml and
# Cargo.lock, which is where maturin reads the package version from.
set -euo pipefail

version="${TAG#v}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.]?[0-9A-Za-z.]+)?$ ]]; then
  echo "Release tag '$TAG' is not a version such as v1.2.3" >&2
  exit 1
fi

export VERSION="$version"
perl -0pi -e 's/^version = "[^"]*"/version = "$ENV{VERSION}"/m' Cargo.toml
perl -0pi -e 's/(name = "fast-walk"\r?\nversion = ")[^"]*/$1$ENV{VERSION}/' Cargo.lock

grep -q "^version = \"$version\"" Cargo.toml
grep -q "^version = \"$version\"" Cargo.lock
echo "Building fast-walk $version"
