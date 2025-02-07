# syntax=docker/dockerfile:1

ARG RUST_VERSION=1.84.0

FROM rust:${RUST_VERSION} AS build
WORKDIR /app

RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y clang lld musl-dev git gcc-x86-64-linux-gnu pkg-config

RUN --mount=type=bind,source=src,target=src \
    --mount=type=bind,source=Cargo.toml,target=Cargo.toml \
    --mount=type=bind,source=Cargo.lock,target=Cargo.lock \
    --mount=type=cache,target=/app/target/ \
    --mount=type=cache,target=/usr/local/cargo/git/db \
    --mount=type=cache,target=/usr/local/cargo/registry/ \
cargo build --locked --release && \
cp ./target/release/neural_network /bin/server


FROM gcr.io/distroless/cc-debian12 AS final

COPY --from=build /bin/server /bin/server
COPY resources/models/20240906081430/network /20240906081430

ENV ROCKET_ADDRESS=0.0.0.0

EXPOSE 8000

CMD ["/bin/server"]