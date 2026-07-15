#ifndef TOKEN_TIMING_HPP
#define TOKEN_TIMING_HPP

#include <string>

struct DllamaTokenNodeTiming {
    unsigned int nodeIndex = 0u;
    unsigned int stageIndex = 0u;
    bool hasStage = false;
    unsigned long long execUs = 0ull;
    unsigned long long syncUs = 0ull;
    unsigned long long bubbleUs = 0ull;
};

bool dllamaTokenTimingPrintEnabled();
std::string formatTokenE2eTimingLine(unsigned int pos, int token, double wallMs);
std::string formatTokenNodeTimingLine(unsigned int pos, const DllamaTokenNodeTiming &timing);

#endif
