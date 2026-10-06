#ifndef TERMINAL_UI_HPP
#define TERMINAL_UI_HPP

#include <iosfwd>
#include <string>
#include <vector>

struct EdgeVisorUiDevice {
    std::string name;
    bool live = false;
    bool inPipeline = false;
    bool offline = false;
};

struct EdgeVisorUiStage {
    std::string name;
    unsigned layerBegin = 0;
    unsigned layerEnd = 0;
    bool computing = false;
    bool degraded = false;
    bool offline = false;
    std::string shadow;
};

void edgeVisorUiEnable(bool enabled);
bool edgeVisorUiEnabled();
void edgeVisorUiPublish(
    const std::vector<EdgeVisorUiDevice> &devices,
    const std::vector<EdgeVisorUiStage> &stages,
    const char *event);
void edgeVisorUiMarkDegraded(unsigned stageIndex, bool degraded);
void edgeVisorUiMarkOffline(unsigned stageIndex, const char *event);
void edgeVisorUiBeginMove(unsigned fromStage, unsigned toStage, const char *event);
void edgeVisorUiShiftLayers(
    unsigned fromStage,
    unsigned toStage,
    unsigned layerBegin,
    unsigned layerEnd,
    const char *event);
void edgeVisorUiEndMove();
void edgeVisorUiSetEvent(const char *event);
void edgeVisorUiOpen();
bool edgeVisorUiIsOpen();
void edgeVisorUiWrite(const char *text);
void edgeVisorUiClose();
void edgeVisorUiRenderPlain(std::ostream &out);

#endif
