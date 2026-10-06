#include "terminal_ui.hpp"

#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

static EdgeVisorUiDevice dev(const char *name, bool live, bool inPipeline, bool offline) {
    EdgeVisorUiDevice device;
    device.name = name;
    device.live = live;
    device.inPipeline = inPipeline;
    device.offline = offline;
    return device;
}

static EdgeVisorUiStage stage(const char *name, unsigned begin, unsigned end, bool computing, bool degraded, bool offline, const char *shadow) {
    EdgeVisorUiStage row;
    row.name = name;
    row.layerBegin = begin;
    row.layerEnd = end;
    row.computing = computing;
    row.degraded = degraded;
    row.offline = offline;
    row.shadow = shadow == nullptr ? "" : shadow;
    return row;
}

static std::vector<EdgeVisorUiDevice> devicesFor(int frame) {
    std::vector<EdgeVisorUiDevice> devices;
    if (frame == 1) {
        devices.push_back(dev("nx1", true, true, false));
        devices.push_back(dev("nx2", false, false, true));
        devices.push_back(dev("laptop", false, false, false));
        devices.push_back(dev("nano2", false, false, false));
        devices.push_back(dev("nano1", false, false, false));
        return devices;
    }
    if (frame == 2) {
        devices.push_back(dev("nx1", true, true, false));
        devices.push_back(dev("nx2", true, true, false));
        devices.push_back(dev("laptop", true, true, false));
        devices.push_back(dev("nano2", false, false, false));
        devices.push_back(dev("nano1", false, false, false));
        return devices;
    }
    devices.push_back(dev("nx1", true, true, false));
    devices.push_back(dev("nx2", true, true, false));
    devices.push_back(dev("laptop", false, false, false));
    devices.push_back(dev("nano2", false, false, false));
    devices.push_back(dev("nano1", false, false, false));
    return devices;
}

static std::vector<EdgeVisorUiStage> stagesFor(int frame) {
    std::vector<EdgeVisorUiStage> stages;
    if (frame == 1) {
        stages.push_back(stage("nx1", 0u, 22u, true, false, false, ""));
        stages.push_back(stage("nx2", 22u, 28u, false, false, true, ""));
    } else if (frame == 2) {
        stages.push_back(stage("nx1", 0u, 22u, true, false, false, ""));
        stages.push_back(stage("nx2", 22u, 24u, true, false, false, ""));
        stages.push_back(stage("laptop", 24u, 28u, true, false, false, ""));
        stages.push_back(stage("nano2", 0u, 0u, false, false, false, ""));
        stages.push_back(stage("nano1", 0u, 0u, false, false, false, ""));
        return stages;
    } else if (frame == 4) {
        stages.push_back(stage("nx1", 0u, 24u, true, false, false, ""));
        stages.push_back(stage("nx2", 24u, 28u, true, true, false, ""));
    } else if (frame == 5) {
        stages.push_back(stage("nx1", 0u, 22u, true, false, false, "L22-26"));
        stages.push_back(stage("nx2", 22u, 28u, true, false, false, "L1-21"));
    } else {
        stages.push_back(stage("nx1", 0u, 22u, true, false, false, ""));
        stages.push_back(stage("nx2", 22u, 28u, true, frame == 3, false, ""));
    }
    stages.push_back(stage("laptop", 0u, 0u, false, false, false, ""));
    stages.push_back(stage("nano2", 0u, 0u, false, false, false, ""));
    stages.push_back(stage("nano1", 0u, 0u, false, false, false, ""));
    return stages;
}

static const char *eventFor(int frame) {
    if (frame <= 0) return "pool ready, three devices not in the pipeline";
    if (frame == 1) return "nx2 offline";
    if (frame == 2) return "laptop joined the pipeline";
    if (frame == 3) return "nx2 degraded";
    if (frame == 4) return "route 1->0  layers 22-23  -51.7 ms";
    return "shadow kv  0->1  5,21";
}

static void showFrame(int frame) {
    edgeVisorUiPublish(devicesFor(frame), stagesFor(frame), eventFor(frame));
    if (frame == 1) edgeVisorUiMarkOffline(1u, eventFor(frame));
    if (frame == 3) edgeVisorUiMarkDegraded(1u, true);
    if (frame == 4) edgeVisorUiBeginMove(1u, 0u, eventFor(frame));
}

int main(int argc, char **argv) {
    const bool plain = argc > 1 && std::strcmp(argv[1], "--plain") == 0;
    edgeVisorUiEnable(true);
    const char *words[] = {"The capital of ", "France is ", "Paris", ", and ", "it sits ", "on the Seine.\n"};
    if (plain) {
        for (int frame = 0; frame < 6; ++frame) {
            showFrame(frame);
            edgeVisorUiRenderPlain(std::cout);
            std::cout << words[frame] << "\n";
        }
        return 0;
    }
    const int only = argc > 1 ? std::atoi(argv[1]) : -1;
    edgeVisorUiOpen();
    const int begin = only >= 0 ? only : 0;
    const int end = only >= 0 ? only + 1 : 6;
    for (int frame = begin; frame < end; ++frame) {
        showFrame(frame);
        edgeVisorUiWrite(words[frame]);
        if (only >= 0) sleep(8);
        else sleep(2);
    }
    if (only < 0) sleep(2);
    edgeVisorUiClose();
    return 0;
}
