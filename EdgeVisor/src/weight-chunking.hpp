#ifndef WEIGHT_CHUNKING_HPP
#define WEIGHT_CHUNKING_HPP

#include <algorithm>
#include <cstddef>

static const size_t NN_WEIGHT_UPLOAD_CHUNK_BYTES = 64u * 1024u * 1024u;

inline size_t nnWeightChunkRows(size_t remainingRows, size_t bytesPerRow, size_t maxChunkBytes) {
    if (remainingRows == 0u || bytesPerRow == 0u) return 0u;
    const size_t rowsByLimit = maxChunkBytes / bytesPerRow;
    return std::min(remainingRows, rowsByLimit == 0u ? 1u : rowsByLimit);
}

#endif
