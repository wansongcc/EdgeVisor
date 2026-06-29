#ifndef DLLAMA_IO_PROFILE_HPP
#define DLLAMA_IO_PROFILE_HPP

#include <cstdint>

enum DllamaIoProbeDirection {
    DLLAMA_IO_PROBE_H2D = 0,
    DLLAMA_IO_PROBE_D2H = 1,
};

void dllamaIoProbeConfigure(const char *logPath);
void dllamaIoProbeConfigureFromEnv();
void dllamaIoProbeSetNode(unsigned int nodeIndex, unsigned int stageIndex);
bool dllamaIoProbeEnabled();
std::uint64_t dllamaIoProbeNowUs();
void dllamaIoProbeFlush(const char *reason);

void dllamaIoProbeRecordNetWriteWall(std::uint64_t elapsedUs);
void dllamaIoProbeRecordNetReadWall(std::uint64_t elapsedUs);
void dllamaIoProbeRecordNetSendSyscall(std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordNetRecvSyscall(std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordNetSendEagain();
void dllamaIoProbeRecordNetRecvEagain();

void dllamaIoProbeRecordHostMemcpy(DllamaIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordVulkanFlush(std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordVulkanInvalidate(std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordVulkanCopyCommand(DllamaIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordCudaMemcpyAsync(DllamaIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes);
void dllamaIoProbeRecordCudaStreamSync(DllamaIoProbeDirection direction, std::uint64_t elapsedUs);

#endif
