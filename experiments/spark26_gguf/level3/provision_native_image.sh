#!/bin/bash
# Trusted operator lane; never invoked from an Agent or candidate environment.
set -euo pipefail
umask 077
[[ $(hostname) == spark-c3bc && $(uname -m) == aarch64 && $(id -u) == 0 ]]
[[ $# == 2 || $# == 3 || $# == 4 ]]
root=$1
archive=$2
inputs=()
if [[ $# -ge 3 ]]; then
    [[ $3 == /* && -d $3 && ! -L $3 ]]
    inputs=(--archive-cache "$3")
fi
vendor_pin=${4:-}
if [[ $# == 4 ]]; then [[ $vendor_pin =~ ^[a-f0-9]{64}$ ]]; fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
[[ $root =~ ^/var/tmp/ee-l3-native\.[A-Za-z0-9]+$ && ! -e $root && ! -L $root ]]
[[ $archive == /* && -f $archive && ! -L $archive ]]
printf '%s  %s\n' e3a71cd04d9513f47e25ffa91644b130b4b20adf0b21554a0b42a78b5433c4ae "$archive" | sha256sum -c
docker() { env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C DOCKER_CONFIG=/dev/null /usr/bin/docker --host unix:///var/run/docker.sock "$@"; }
base=evolveengine/spark26-q8-diagnostic-e0:x4xmpm
base_id=sha256:28e47ed30260eaaab0fee0251b53018766afc2dc61e1c8e4c65be4b1e6d088ff
[[ $(docker image inspect --format '{{.Id}}' "$base") == "$base_id" ]]
mkdir "$root"
mkdir "$root/stage"
exec > >(tee "$root/provision.log") 2>&1
printf 'stage_root=%s\n' "$root"
timeout --signal=TERM --kill-after=10 360 python3 -I "$here/native_image_inputs.py" "$root/context" "${inputs[@]}"
tar -xzf "$archive" --no-same-owner --no-same-permissions --strip-components=1 -C "$root/stage"
suffix=${root##*.}
name=ee-native-provision-${suffix,,}
clean() {
  docker rm -f "$name" >/dev/null 2>&1 || :
  if docker inspect "$name" >/dev/null 2>&1; then
    echo 'Native staging container cleanup not verified' >&2; exit 9
  fi
}
if [[ -n $vendor_pin ]]; then
    # The trusted operator generated this cache using cargo vendor --locked
    # against the exact E0 archive and pins the transport bytes independently.
    cache=$3
    compiler=$root/stage/rustup/toolchains/1.92.0-aarch64-unknown-linux-gnu
    mkdir -p "$root/stage/packed-cargo" "$compiler" "$root/stage/components"
    cat >"$root/stage/rust-pins.sha256" <<'PINS'
7c8706fad4c038b5eacab0092e15db54d2b365d5f3323ca046fe987f814e7826  rustc-1.92.0-aarch64-unknown-linux-gnu.tar.xz
ce2ab42c09d633b0a8b4b65a297c700ae0fad47aae890f75894782f95be7e36d  rust-std-1.92.0-aarch64-unknown-linux-gnu.tar.xz
cb2ce6be6411b986e25c71ad8a813f9dfbe3461738136fd684e3644f8dd75df4  cargo-1.92.0-aarch64-unknown-linux-gnu.tar.xz
PINS
    printf '%s  %s\n' "$vendor_pin" vendor-cache.tar.gz >>"$root/stage/rust-pins.sha256"
    (cd "$cache"; sha256sum -c "$root/stage/rust-pins.sha256")
    while read -r pin file; do
        [[ $file == *.tar.xz ]] || continue
        tar -xJf "$cache/$file" --no-same-owner --no-same-permissions -C "$root/stage/components"
    done <"$root/stage/rust-pins.sha256"
    tar -xzf "$cache/vendor-cache.tar.gz" --no-same-owner --no-same-permissions -C "$root/stage/packed-cargo"
    trap clean EXIT
    docker create --name "$name" --pull=never --platform linux/arm64 --runtime=runc --network none \
        --read-only --cpus 4 --memory 8g --memory-swap 8g --pids-limit 128 --security-opt no-new-privileges \
        --cap-drop ALL --no-healthcheck --restart no --log-driver local --log-opt max-size=8m \
        --tmpfs /tmp:rw,nosuid,nodev,size=1g --mount "type=bind,src=$root/stage,dst=/stage" \
        --env NVIDIA_VISIBLE_DEVICES=void --env HOME=/stage --entrypoint /bin/bash "$base_id" -euc '
            prefix=/stage/rustup/toolchains/1.92.0-aarch64-unknown-linux-gnu
            for script in /stage/components/*/install.sh; do
                /bin/bash "$script" --prefix="$prefix" --disable-ldconfig
            done
            export PATH="$prefix/bin:/usr/bin:/bin" CARGO_HOME=/stage/cargo-check
            rustc --version | grep "^rustc 1\.92\.0 "
            cargo --version | grep "^cargo 1\.92\.0 "
            mkdir "$CARGO_HOME"
            printf "[source.crates-io]\nreplace-with = \"vendor\"\n[source.vendor]\ndirectory = \"/stage/packed-cargo/vendor\"\n" >"$CARGO_HOME/config.toml"
            cd /stage/rust
            cargo metadata --locked --offline --format-version 1 --filter-platform aarch64-unknown-linux-gnu >/stage/cargo.metadata.json
            cargo metadata --manifest-path sglang-radix-tree/Cargo.toml --locked --offline --format-version 1 --filter-platform aarch64-unknown-linux-gnu >/stage/cargo.radix.metadata.json
        '
else
    trap clean EXIT
    docker create --name "$name" --pull=never --platform linux/arm64 --runtime=runc --network bridge \
    --read-only --cpus 4 --memory 8g --memory-swap 8g --pids-limit 128 --security-opt no-new-privileges \
    --cap-drop ALL --no-healthcheck --restart no --log-driver local --log-opt max-size=8m \
    --tmpfs /tmp:rw,nosuid,nodev,size=1g --mount "type=bind,src=$root/stage,dst=/stage" \
    --env NVIDIA_VISIBLE_DEVICES=void --env HOME=/stage/provision-home --env RUSTUP_HOME=/stage/rustup \
    --env CARGO_HOME=/stage/provision-home/.cargo --env CARGO_NET_RETRY=1 --env CARGO_HTTP_TIMEOUT=30 \
    --env RUSTUP_DIST_SERVER=https://static.rust-lang.org --entrypoint /bin/bash "$base_id" -euc '
        mkdir /stage/provision-home /stage/rustup /stage/packed-cargo
        /root/.cargo/bin/rustup toolchain install 1.92.0 --profile minimal --no-self-update
        compiler=/stage/rustup/toolchains/1.92.0-aarch64-unknown-linux-gnu/bin
        "$compiler/rustc" --version
        "$compiler/cargo" --version
        export PATH="$compiler:/usr/bin:/bin"
        cd /stage/rust
        "$compiler/cargo" vendor --locked --sync sglang-radix-tree/Cargo.toml --versioned-dirs /stage/packed-cargo/vendor >/stage/cargo.generated.toml
    '
fi
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C DOCKER_CONFIG=/dev/null \
    timeout --signal=TERM --kill-after=15 1050 /usr/bin/docker --host unix:///var/run/docker.sock start -a "$name"
[[ $(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$name") == 'exited 0' ]]
clean
trap - EXIT
[[ $(find "$root/stage/packed-cargo/vendor" -mindepth 1 -maxdepth 1 -type d | wc -l) == 585 ]]
mv "$root/stage/packed-cargo" "$root/context/cargo"
mv "$root/stage/rustup/toolchains/1.92.0-aarch64-unknown-linux-gnu" "$root/context/rust-toolchain"
cat >"$root/context/cargo/config.toml" <<'TOML'
[source.crates-io]
replace-with = "vendored-sources"
[source.vendored-sources]
directory = "/opt/ee-native/cargo/vendor"
[net]
offline = true
TOML
if [[ -n $vendor_pin ]]; then
    install -m 644 "$root/stage/rust-pins.sha256" "$root/context/cargo/rust-pins.sha256"
fi
install -m 644 "$here/Dockerfile.native" "$root/context/Dockerfile"
printf '/downloads/\n' >"$root/context/.dockerignore"
chmod 755 "$root/context/cargo" "$root/context/cargo/vendor"
chmod 644 "$root/context/cargo/config.toml"
# COPY retains modes. The resources, but never the Docker context, must be
# readable by the build UID in the network-disabled candidate container.
chmod -R a+rX "$root/context/cargo" "$root/context/rust-toolchain" "$root/context/sources" "$root/context/wheels"
chmod 644 "$root/context/input-manifest.json"
img=evolveengine/local-level3-native:${suffix,,}
[[ $(docker image inspect --format '{{.Id}}' "$base") == "$base_id" ]]
env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin LC_ALL=C DOCKER_CONFIG=/dev/null DOCKER_BUILDKIT=0 \
    timeout --signal=TERM --kill-after=15 900 /usr/bin/docker --host unix:///var/run/docker.sock build \
    --network none --pull=false --build-arg "BASE=$base" --tag "$img" "$root/context"
[[ $(docker image inspect --format '{{.Id}}' "$base") == "$base_id" ]]
id=$(docker image inspect --format '{{.Id}}' "$img")
[[ $id =~ ^sha256:[a-f0-9]{64}$ ]]
printf '%s\n' "$id" >"$root/image.id"
sha256sum "$root/context/input-manifest.json" "$archive" "$root/context/Dockerfile" | tee "$root/pins.sha256"
