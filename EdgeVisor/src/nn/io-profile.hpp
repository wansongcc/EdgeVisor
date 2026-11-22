#ifndef EDGEVISOR_IO_PROFILE_HPP
#define EDGEVISOR_IO_PROFILE_HPP

#include <cstdint>

enum EdgeVisorIoProbeDirection {
    EDGEVISOR_IO_PROBE_H2D = 0,
    EDGEVISOR_IO_PROBE_D2H = 1,
};

void edgevisorIoProbeConfigure(const char *logPath);
void edgevisorIoProbeConfigureFromEnv();
void edgevisorIoProbeSetNode(unsigned int nodeIndex, unsigned int stageIndex);
bool edgevisorIoProbeEnabled();
std::uint64_t edgevisorIoProbeNowUs();
void edgevisorIoProbeFlush(const char *reason);

void edgevisorIoProbeRecordNetWriteWall(std::uint64_t elapsedUs);
void edgevisorIoProbeRecordNetReadWall(std::uint64_t elapsedUs);
void edgevisorIoProbeRecordNetSendSyscall(std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordNetRecvSyscall(std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordNetSendEagain();
void edgevisorIoProbeRecordNetRecvEagain();

void edgevisorIoProbeRecordHostMemcpy(EdgeVisorIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordVulkanFlush(std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordVulkanInvalidate(std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordVulkanCopyCommand(EdgeVisorIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordCudaMemcpyAsync(EdgeVisorIoProbeDirection direction, std::uint64_t elapsedUs, std::uint64_t bytes);
void edgevisorIoProbeRecordCudaStreamSync(EdgeVisorIoProbeDirection direction, std::uint64_t elapsedUs);

#endif
