# 0002. Cache model weights by file size: IndexedDB buffers, OPFS streams, or a direct URL

Status: accepted (commits `3fb8ac5`, `be5a724`, `9d6bf51`, `a7844d5`, `1227c24`, all 2026-03-23)

## Context

The weights are 316 MB to 2.3 GB, split into ONNX graphs and external data files. They have to persist across visits, and loading them must not exhaust the JS heap. The first version used the Cache API. The commit message of `3fb8ac5` says the Cache API failed silently on large writes in Safari, especially on iOS. According to the message of `9d6bf51`, the first IndexedDB version peaked at about 1.1 GB of memory for a 221 MB file. That came from copies through a chunks array, the merged buffer, a Blob, an IDB clone, and the caller's buffer.

## Decision

In `src/vl-model.js`:

- Files up to `LARGE_FILE_THRESHOLD` (2 GB on desktop, 50 MB on mobile) are streamed into one pre-allocated `ArrayBuffer` sized from `content-length`. The same buffer goes to ONNX Runtime and to IndexedDB, with the IndexedDB write left unawaited.
- Larger files are streamed straight into OPFS through `createWritable()` and passed to ONNX Runtime as a blob URL. If OPFS is unavailable, the remote URL is passed instead, and ONNX Runtime fetches it itself.
- Each entry stores its expected size, and incomplete entries are downloaded again (`be5a724`).

## Consequences

- Large files no longer have to be held in JS memory when OPFS is available. Desktop loads after the first visit read from IndexedDB.
- There are three code paths to keep correct, and none has automated tests because each needs a real browser.
- The OPFS cache-hit check sends a `HEAD` request, so a cached large file still needs the network to confirm its size.
- Without OPFS, files over the threshold are not cached and are downloaded again on each visit.

## Alternatives considered

- The Cache API: replaced because of the Safari failures in `3fb8ac5`. `clearModelCache` still deletes the old `onnx-models-v1` bucket.
- Always passing URLs to ONNX Runtime (`a7844d5`): this keeps JS memory low but loses persistence. OPFS was added on top in `1227c24`.
