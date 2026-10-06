#include "terminal_ui.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <sys/ioctl.h>

namespace {

std::mutex g_mu;
bool g_enabled = false;
bool g_open = false;
bool g_tty = false;
int g_rows = 24;
int g_cols = 80;
int g_panelRows = 12;
int g_moveFrom = -1;
int g_moveTo = -1;
std::vector<EdgeVisorUiDevice> g_devices;
std::vector<EdgeVisorUiStage> g_stages;
std::string g_event = "waiting for devices";

enum Color {
    COL_OFF = 0,
    COL_CYAN,
    COL_GREEN,
    COL_YELLOW,
    COL_DIM,
    COL_WHITE,
    COL_RED,
    COL_BLUE,
    COL_MAGENTA
};

struct Seg {
    std::string text;
    Color color;
};

struct Line {
    std::vector<Seg> segs;
};

const char *colorCode(Color color) {
    switch (color) {
        case COL_CYAN: return "\033[0;1;36m";
        case COL_GREEN: return "\033[0;1;32m";
        case COL_YELLOW: return "\033[0;1;33m";
        case COL_DIM: return "\033[0;2;37m";
        case COL_WHITE: return "\033[0;1;37m";
        case COL_RED: return "\033[0;1;31m";
        case COL_BLUE: return "\033[0;1;34m";
        case COL_MAGENTA: return "\033[0;1;35m";
        default: return "\033[0m";
    }
}

void readTermSize() {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        g_cols = (int)ws.ws_col;
        g_rows = (int)ws.ws_row;
    }
}

std::string fit(const std::string &text, size_t width) {
    if (text.size() >= width) return text.substr(0, width);
    return text + std::string(width - text.size(), ' ');
}

std::string lineText(const Line &line) {
    std::string out;
    for (size_t i = 0; i < line.segs.size(); ++i) out += line.segs[i].text;
    return out;
}

void add(Line &line, const std::string &text, Color color) {
    line.segs.push_back(Seg{text, color});
}

void closeBar(Line &line, int width) {
    const size_t used = lineText(line).size();
    if (used + 1 < (size_t)width) add(line, std::string((size_t)width - used - 1, ' '), COL_WHITE);
    add(line, "|", COL_WHITE);
}

Line textLine(const std::string &text, Color color, int width) {
    Line line;
    add(line, "| ", COL_WHITE);
    std::string body = text;
    const size_t room = (size_t)width > 3 ? (size_t)width - 3 : 0;
    if (body.size() > room) body = body.substr(0, room);
    add(line, body, color);
    closeBar(line, width);
    return line;
}

std::string border(int width, const std::string &title) {
    if (width < 4) return std::string((size_t)width, '-');
    std::string body = title.empty() ? "" : " " + title + " ";
    if (body.size() > (size_t)width - 2) body = body.substr(0, (size_t)width - 2);
    const size_t dashes = (size_t)width - 2 - body.size();
    const size_t left = title.empty() ? dashes : 2;
    const size_t right = dashes > left ? dashes - left : 0;
    return "+" + std::string(left, '-') + body + std::string(right, '-') + "+";
}

bool stageDegraded(size_t index) {
    return index < g_stages.size() && g_stages[index].degraded;
}

bool nameDegraded(const std::string &name) {
    for (size_t i = 0; i < g_stages.size(); ++i) {
        if (g_stages[i].name == name && (g_stages[i].degraded || (int)i == g_moveFrom)) return true;
    }
    return false;
}

Color stageColor(size_t index) {
    if (index >= g_stages.size()) return COL_DIM;
    const EdgeVisorUiStage &stage = g_stages[index];
    if (stage.offline) return COL_BLUE;
    if (!stage.computing || stage.layerEnd <= stage.layerBegin) return COL_DIM;
    if ((int)index == g_moveFrom || stage.degraded) return COL_RED;
    if ((int)index == g_moveTo) return COL_GREEN;
    return COL_CYAN;
}

std::string boxEdge(bool solid, size_t inner) {
    std::string edge = "+";
    for (size_t i = 0; i < inner; ++i) edge.push_back(solid ? '-' : ((i % 2) == 0 ? ' ' : '-'));
    edge.push_back('+');
    return edge;
}

std::vector<Line> layout(int cols) {
    const int width = cols < 48 ? 48 : cols;
    std::vector<Line> lines;
    lines.push_back(textLine(border(width, "EdgeVisor").substr(1, (size_t)width - 2), COL_CYAN, width));
    lines[0] = Line();
    add(lines[0], border(width, "EdgeVisor"), COL_CYAN);

    lines.push_back(textLine("device pool", COL_WHITE, width));

    Line pool;
    add(pool, "| ", COL_WHITE);
    if (g_devices.empty()) add(pool, "(empty)", COL_DIM);
    for (size_t i = 0; i < g_devices.size(); ++i) {
        if (lineText(pool).size() > (size_t)width - 18) break;
        if (i > 0) add(pool, "  ", COL_WHITE);
        const bool offline = g_devices[i].offline;
        const bool reserved = !offline && !g_devices[i].live;
        const bool slow = !offline && nameDegraded(g_devices[i].name);
        const char *state = offline ? "off" : (reserved ? "reserved" : (slow ? "slow" : (g_devices[i].inPipeline ? "live" : "idle")));
        const Color chip = offline ? COL_BLUE : (reserved ? COL_YELLOW : (slow ? COL_RED : (g_devices[i].inPipeline ? COL_GREEN : COL_DIM)));
        add(pool, "[" + g_devices[i].name + " " + state + "]", chip);
    }
    closeBar(pool, width);
    lines.push_back(pool);

    Line bar;
    add(bar, "| slow ", COL_WHITE);
    for (int i = 0; i < 24; ++i) {
        const Color block = i < 8 ? COL_RED : (i < 16 ? COL_YELLOW : COL_GREEN);
        add(bar, "#", block);
    }
    const int liveCount = (int)std::count_if(g_devices.begin(), g_devices.end(), [](const EdgeVisorUiDevice &d) {
        return d.live && !d.offline;
    });
    std::ostringstream tail;
    tail << " fast   " << liveCount << "/" << g_devices.size() << " up";
    add(bar, tail.str(), COL_WHITE);
    closeBar(bar, width);
    lines.push_back(bar);
    lines.push_back(textLine("pipeline", COL_WHITE, width));

    const size_t n = g_stages.size();
    size_t inner = 12;
    size_t arrowWidth = 5;
    if (n > 1) {
        while (n * (inner + 2) + (n - 1) * arrowWidth + 2 > (size_t)width && inner > 8) inner--;
        if (n * (inner + 2) + (n - 1) * arrowWidth + 2 > (size_t)width) arrowWidth = 3;
    }
    Line tops;
    Line names;
    Line layers;
    Line shadows;
    Line bots;
    add(tops, "|", COL_WHITE);
    add(names, "|", COL_WHITE);
    add(layers, "|", COL_WHITE);
    add(shadows, "|", COL_WHITE);
    add(bots, "|", COL_WHITE);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            const bool hot = (g_moveFrom == (int)i - 1 && g_moveTo == (int)i) ||
                (g_moveFrom == (int)i && g_moveTo == (int)i - 1);
            std::string gap(arrowWidth, ' ');
            if (hot) {
                if (arrowWidth >= 5) gap = g_moveTo > g_moveFrom ? " ==>" : " <==";
                else gap = g_moveTo > g_moveFrom ? " >" : " <";
                if (gap.size() < arrowWidth) gap += std::string(arrowWidth - gap.size(), ' ');
            }
            const Color gapColor = hot ? COL_RED : COL_DIM;
            add(tops, std::string(arrowWidth, ' '), COL_DIM);
            add(names, gap, gapColor);
            add(layers, gap, gapColor);
            add(shadows, std::string(arrowWidth, ' '), COL_DIM);
            add(bots, std::string(arrowWidth, ' '), COL_DIM);
        }
        const bool on = g_stages[i].computing && g_stages[i].layerEnd > g_stages[i].layerBegin;
        const Color color = stageColor(i);
        add(tops, boxEdge(on, inner), color);
        std::string name = " " + g_stages[i].name;
        if (g_stages[i].offline) name += " OFF";
        else if (g_stages[i].degraded || (int)i == g_moveFrom) name += " SLOW";
        else if ((int)i == g_moveTo) name += " IN";
        add(names, "|" + fit(name, inner) + "|", color);
        std::string label = " waiting";
        if ((on || g_stages[i].offline) && g_stages[i].layerEnd > g_stages[i].layerBegin) {
            std::ostringstream lab;
            lab << " L" << g_stages[i].layerBegin << "-" << (g_stages[i].layerEnd - 1u);
            label = lab.str();
        }
        add(layers, "|" + fit(label, inner) + "|", color);
        std::string shadow = g_stages[i].shadow.empty() ? " " : (" sh " + g_stages[i].shadow);
        add(shadows, "|" + fit(shadow, inner) + "|", g_stages[i].shadow.empty() ? color : COL_MAGENTA);
        add(bots, boxEdge(on, inner), color);
    }
    closeBar(tops, width);
    closeBar(names, width);
    closeBar(layers, width);
    closeBar(shadows, width);
    closeBar(bots, width);
    lines.push_back(tops);
    lines.push_back(names);
    lines.push_back(layers);
    lines.push_back(shadows);
    lines.push_back(bots);

    Color eventColor = COL_WHITE;
    if (g_moveFrom >= 0) eventColor = COL_RED;
    else if (g_event.find("offline") != std::string::npos) eventColor = COL_BLUE;
    else if (g_event.find("shadow") != std::string::npos) eventColor = COL_MAGENTA;
    else if (g_event.find("route") != std::string::npos) eventColor = COL_YELLOW;
    else if (g_event.find("joined") != std::string::npos) eventColor = COL_GREEN;
    else if (g_event.find("degraded") != std::string::npos) eventColor = COL_RED;
    lines.push_back(textLine(g_event.empty() ? " " : g_event, eventColor, width));
    Line bottom;
    add(bottom, border(width, ""), COL_CYAN);
    lines.push_back(bottom);
    return lines;
}

void paintLocked() {
    const std::vector<Line> lines = layout(g_cols);
    g_panelRows = (int)lines.size();
    if (g_rows < g_panelRows + 4) g_rows = g_panelRows + 4;
    std::fputs("\0337", stdout);
    std::fputs("\033[r", stdout);
    for (int i = 0; i < g_panelRows; ++i) {
        std::fprintf(stdout, "\033[%d;1H\033[2K", i + 1);
        if (i < (int)lines.size()) {
            const Line &line = lines[(size_t)i];
            for (size_t s = 0; s < line.segs.size(); ++s) {
                std::fputs(colorCode(line.segs[s].color), stdout);
                std::fputs(line.segs[s].text.c_str(), stdout);
            }
            std::fputs("\033[0m", stdout);
        }
    }
    std::fprintf(stdout, "\033[%d;%dr", g_panelRows + 1, g_rows);
    std::fputs("\0338", stdout);
    std::fflush(stdout);
}

void restoreTermLocked() {
    if (!g_tty) return;
    std::fputs("\033[r\033[0m", stdout);
    std::fprintf(stdout, "\033[%d;1H\n", g_rows > 0 ? g_rows : 24);
    std::fflush(stdout);
}

void redrawLocked() {
    if (g_open && g_tty) paintLocked();
}

} // namespace

void edgeVisorUiEnable(bool enabled) { g_enabled = enabled; }
bool edgeVisorUiEnabled() { return g_enabled; }
bool edgeVisorUiIsOpen() { return g_open; }

void edgeVisorUiPublish(
    const std::vector<EdgeVisorUiDevice> &devices,
    const std::vector<EdgeVisorUiStage> &stages,
    const char *event) {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_devices = devices;
    g_stages = stages;
    g_moveFrom = -1;
    g_moveTo = -1;
    if (event != nullptr && event[0] != '\0') g_event = event;
    redrawLocked();
}

void edgeVisorUiMarkDegraded(unsigned stageIndex, bool degraded) {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    if (stageIndex < g_stages.size()) g_stages[stageIndex].degraded = degraded;
    if (degraded && (g_event.empty() || g_event == "pipeline set" || g_event.find("pool") != std::string::npos)) {
        std::ostringstream oss;
        oss << "stage " << stageIndex << " degraded";
        g_event = oss.str();
    }
    redrawLocked();
}

void edgeVisorUiMarkOffline(unsigned stageIndex, const char *event) {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    if (stageIndex < g_stages.size()) {
        EdgeVisorUiStage &stage = g_stages[stageIndex];
        stage.offline = true;
        stage.computing = false;
        stage.degraded = false;
        for (size_t i = 0; i < g_devices.size(); ++i) {
            if (g_devices[i].name == stage.name) {
                g_devices[i].offline = true;
                g_devices[i].live = false;
                g_devices[i].inPipeline = false;
            }
        }
        if (event == nullptr || event[0] == '\0') g_event = stage.name + " offline";
    }
    if (event != nullptr && event[0] != '\0') g_event = event;
    redrawLocked();
}

void edgeVisorUiBeginMove(unsigned fromStage, unsigned toStage, const char *event) {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_moveFrom = (int)fromStage;
    g_moveTo = (int)toStage;
    if (fromStage < g_stages.size()) g_stages[fromStage].degraded = true;
    if (event != nullptr && event[0] != '\0') g_event = event;
    redrawLocked();
}

void edgeVisorUiShiftLayers(
    unsigned fromStage,
    unsigned toStage,
    unsigned layerBegin,
    unsigned layerEnd,
    const char *event) {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_moveFrom = (int)fromStage;
    g_moveTo = (int)toStage;
    if (fromStage < g_stages.size() && toStage < g_stages.size() && layerEnd > layerBegin) {
        EdgeVisorUiStage &src = g_stages[fromStage];
        EdgeVisorUiStage &dst = g_stages[toStage];
        src.degraded = true;
        if (toStage > fromStage) {
            if (layerBegin >= src.layerBegin && layerBegin <= src.layerEnd) src.layerEnd = layerBegin;
            if (layerBegin < dst.layerBegin) dst.layerBegin = layerBegin;
        } else if (fromStage > toStage) {
            if (layerEnd <= src.layerEnd && layerEnd >= src.layerBegin) src.layerBegin = layerEnd;
            if (layerEnd > dst.layerEnd) dst.layerEnd = layerEnd;
        }
        src.computing = src.layerEnd > src.layerBegin;
        dst.computing = dst.layerEnd > dst.layerBegin;
    }
    if (event != nullptr && event[0] != '\0') g_event = event;
    redrawLocked();
}

void edgeVisorUiEndMove() {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_moveFrom = -1;
    g_moveTo = -1;
    redrawLocked();
}

void edgeVisorUiSetEvent(const char *event) {
    if (!g_enabled) return;
    std::lock_guard<std::mutex> lock(g_mu);
    if (event != nullptr) g_event = event;
    redrawLocked();
}

void edgeVisorUiOpen() {
    if (!g_enabled || g_open) return;
    std::lock_guard<std::mutex> lock(g_mu);
    g_tty = isatty(STDOUT_FILENO) == 1;
    readTermSize();
    if (!g_tty) {
        const std::vector<Line> lines = layout(96);
        for (size_t i = 0; i < lines.size(); ++i) std::cout << lineText(lines[i]) << "\n";
        g_open = true;
        return;
    }
    std::fputs("\033[2J\033[H\033[?25h", stdout);
    const std::vector<Line> lines = layout(g_cols);
    g_panelRows = (int)lines.size();
    for (int i = 0; i < g_panelRows; ++i) {
        const Line &line = lines[(size_t)i];
        for (size_t s = 0; s < line.segs.size(); ++s) {
            std::fputs(colorCode(line.segs[s].color), stdout);
            std::fputs(line.segs[s].text.c_str(), stdout);
        }
        std::fputs("\033[0m\033[K\n", stdout);
    }
    if (g_rows <= g_panelRows + 1) g_rows = g_panelRows + 6;
    std::fprintf(stdout, "\033[%d;%dr", g_panelRows + 1, g_rows);
    std::fprintf(stdout, "\033[%d;1H", g_panelRows + 1);
    std::fflush(stdout);
    g_open = true;
    static bool hooked = false;
    if (!hooked) {
        std::atexit(edgeVisorUiClose);
        hooked = true;
    }
}

void edgeVisorUiWrite(const char *text) {
    if (text == nullptr) return;
    if (!g_open) {
        std::fputs(text, stdout);
        std::fflush(stdout);
        return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    std::fputs(text, stdout);
    std::fflush(stdout);
}

void edgeVisorUiClose() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_open) return;
    restoreTermLocked();
    g_open = false;
}

void edgeVisorUiRenderPlain(std::ostream &out) {
    const std::vector<Line> lines = layout(96);
    for (size_t i = 0; i < lines.size(); ++i) out << lineText(lines[i]) << "\n";
}
