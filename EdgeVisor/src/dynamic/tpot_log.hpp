#pragma once

#include "dynamic/tpot_algorithm.hpp"

#include <cctype>
#include <sstream>
#include <string>

namespace dllama {
namespace dynamic_tpot {

inline std::string sanitizeCandidateLogValue(const std::string &value) {
    if (value.empty()) return "none";
    std::string out = value;
    for (size_t i = 0u; i < out.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(out[i]);
        if (std::isspace(c) || out[i] == '=') out[i] = '_';
    }
    return out;
}

inline std::string formatPpCandidateLogFields(const Candidate &candidate) {
    std::ostringstream oss;
    oss << "pp_best_valid=" << (candidate.valid ? 1 : 0)
        << " pp_best_gain_ms=" << candidate.gainMs
        << " pp_best_threshold_ms=" << candidate.thresholdMs
        << " pp_best_reason=" << sanitizeCandidateLogValue(candidate.reason)
        << " pp_best_from_stage=" << candidate.fromStageIndex
        << " pp_best_to_stage=" << candidate.toStageIndex
        << " pp_best_layer=" << candidate.layerIndex
        << " pp_best_layer_count=" << candidate.layerCount;
    return oss.str();
}

inline std::string formatSelectedCandidateLogFields(const Candidate &candidate) {
    std::ostringstream oss;
    oss << "best=" << candidateKindName(candidate.kind)
        << " gain_ms=" << candidate.gainMs
        << " threshold_ms=" << candidate.thresholdMs
        << " from_stage=" << candidate.fromStageIndex
        << " to_stage=" << candidate.toStageIndex
        << " from_node=" << candidate.fromNodeIndex
        << " to_node=" << candidate.toNodeIndex
        << " layer=" << candidate.layerIndex
        << " layer_count=" << candidate.layerCount
        << " head_move=" << candidate.headMove
        << " ffn_move=" << candidate.ffnMove;
    return oss.str();
}

} // namespace dynamic_tpot
} // namespace dllama
