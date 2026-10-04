#!/usr/bin/env bash
# 音频编解码工具；FFmpeg、Ogg、Vorbis 源码归档随整合包分发。
set -euo pipefail
source_archive="$1"
build_directory="$2"
source_directory="$(cd "$(dirname "$source_archive")" && pwd)"
mkdir -p "$build_directory"
build_directory="$(cd "$build_directory" && pwd)"
prefix="$build_directory/dependencies"
export PKG_CONFIG_PATH="$prefix/lib/pkgconfig"
for library in libogg-1.3.5 libvorbis-1.3.7; do
  mkdir -p "$build_directory/$library"
  tar -xf "$source_directory/$library.tar.xz" -C "$build_directory/$library" --strip-components=1
  (cd "$build_directory/$library"; ./configure --prefix="$prefix" --disable-shared --enable-static; make -j 3 CFLAGS="-O2 -fPIC"; make install CFLAGS="-O2 -fPIC")
done
tar -xf "$source_archive" -C "$build_directory" --strip-components=1
cd "$build_directory"
extra=(--disable-autodetect)
if [[ "${MSYSTEM:-}" == MINGW64 ]]; then extra+=(--extra-ldflags=-static); fi
./configure "${extra[@]}" --disable-network --disable-doc --disable-debug \
  --disable-asm --disable-everything --enable-small --enable-static --disable-shared \
  --enable-ffmpeg --disable-ffplay --disable-ffprobe --enable-libvorbis --pkg-config-flags=--static \
  --enable-avcodec --enable-avformat --enable-avfilter --enable-swresample \
  --enable-protocol=file,pipe \
  --enable-demuxer=aac,flac,matroska,mov,mp3,ogg,wav,pcm_s16le \
  --enable-decoder=aac,alac,flac,mp3,opus,pcm_s16le,pcm_s24le,pcm_s32le,pcm_f32le,pcm_f64le,vorbis \
  --enable-encoder=pcm_s16le,pcm_s24le,pcm_f32le,aac,libvorbis \
  --enable-muxer=wav,pcm_s16le,pcm_f32le,adts,ogg \
  --enable-parser=aac,flac,mpegaudio,opus,vorbis \
  --enable-filter=aresample,aformat,anull,atrim,asetpts
make -j 3
