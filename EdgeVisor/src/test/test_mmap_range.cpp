#include "mmap.hpp"

#include <assert.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

int main() {
    char path[] = "/tmp/dllama-mmap-range-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);

    unsigned char bytes[8192];
    for (size_t index = 0; index < sizeof(bytes); ++index) bytes[index] = (unsigned char)(index % 251u);
    assert(write(fd, bytes, sizeof(bytes)) == (ssize_t)sizeof(bytes));
    close(fd);

    MmapFile file;
    openMmapFileRange(&file, path, 123u, 4097u);
    assert(file.size == 4097u);
    assert(std::memcmp(file.data, bytes + 123u, file.size) == 0);
    closeMmapFile(&file);

    unlink(path);
    return 0;
}
