#include "weight-chunking.hpp"

#include <cassert>

int main() {
    assert(nnWeightChunkRows(100u, 100u, 250u) == 2u);
    assert(nnWeightChunkRows(1u, 1000u, 250u) == 1u);
    assert(nnWeightChunkRows(0u, 100u, 250u) == 0u);
    return 0;
}
