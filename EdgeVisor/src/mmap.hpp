#ifndef MMAP_HPP
#define MMAP_HPP

#include <cstdio>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

struct MmapFile {
    void* data;
    size_t size;
    void* mappingData;
    size_t mappingSize;
#ifdef _WIN32
    HANDLE hFile;
    HANDLE hMapping;
#else
    int fd;
#endif
};

size_t mmapAllocationGranularity() {
#ifdef _WIN32
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    return (size_t)systemInfo.dwAllocationGranularity;
#else
    long pageSize = sysconf(_SC_PAGE_SIZE);
    return pageSize > 0 ? (size_t)pageSize : 4096u;
#endif
}

long seekToEnd(FILE* file) {
#ifdef _WIN32
    _fseeki64(file, 0, SEEK_END);
    return _ftelli64(file);
#else
    fseek(file, 0, SEEK_END);
    return ftell(file);
#endif
}

void openMmapFileRange(MmapFile *file, const char *path, size_t offset, size_t size) {
    if (size == 0u) throw std::runtime_error("Cannot mmap an empty file range");
    file->size = size;
    const size_t granularity = mmapAllocationGranularity();
    const size_t mappingOffset = offset - offset % granularity;
    const size_t dataOffset = offset - mappingOffset;
    file->mappingSize = size + dataOffset;
#ifdef _WIN32
    file->hFile = CreateFileA(path, GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file->hFile == INVALID_HANDLE_VALUE) {
        printf("Cannot open file %s\n", path);
        exit(EXIT_FAILURE);
    }

    file->hMapping = CreateFileMappingA(file->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (file->hMapping == NULL) {
        printf("CreateFileMappingA failed, error: %lu\n", GetLastError());
        CloseHandle(file->hFile);
        exit(EXIT_FAILURE);
    }

    const DWORD offsetHigh = (DWORD)(mappingOffset >> 32u);
    const DWORD offsetLow = (DWORD)(mappingOffset & 0xffffffffu);
    file->mappingData = (void *)MapViewOfFile(file->hMapping, FILE_MAP_READ,
        offsetHigh, offsetLow, file->mappingSize);
    if (file->mappingData == NULL) {
        printf("MapViewOfFile failed!\n");
        CloseHandle(file->hMapping);
        CloseHandle(file->hFile);
        exit(EXIT_FAILURE);
    }
    file->data = (void *)((unsigned char *)file->mappingData + dataOffset);
#else
    file->fd = open(path, O_RDONLY);
    if (file->fd == -1) {
        throw std::runtime_error("Cannot open file");
    }

    file->mappingData = mmap(NULL, file->mappingSize, PROT_READ, MAP_PRIVATE, file->fd, mappingOffset);
    if (file->mappingData == MAP_FAILED) {
        close(file->fd);
        throw std::runtime_error("Mmap failed");
    }
    file->data = (void *)((unsigned char *)file->mappingData + dataOffset);
#endif
}

void openMmapFile(MmapFile *file, const char *path, size_t size) {
    openMmapFileRange(file, path, 0u, size);
}

void closeMmapFile(MmapFile *file) {
#ifdef _WIN32
    UnmapViewOfFile(file->mappingData);
    CloseHandle(file->hMapping);
    CloseHandle(file->hFile);
#else
    munmap(file->mappingData, file->mappingSize);
    close(file->fd);
#endif
}

#endif
