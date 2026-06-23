// Unit tests exercise device/operator code without constructing an LLM graph.
// Keep runtime feature gates disabled unless a test explicitly supplies them.
bool getEnableStageFullWeights() {
    return false;
}

bool getEnableKvRedundancyDuringMigration() {
    return false;
}

bool getAllowNoShadowHeadMigration() {
    return false;
}
