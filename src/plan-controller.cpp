#include "plan-controller.hpp"

#include "app.hpp"
#include "json.hpp"
#include "plan-command.hpp"

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <errno.h>
#endif

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

using json = nlohmann::json;

static bool readLine(int fd, std::string &out) {
    out.clear();
    char ch;
    while (true) {
        ssize_t n = ::read(fd, &ch, 1);
        if (n == 0) return !out.empty();
        if (n < 0) return false;
        if (ch == '\n') return true;
        out.push_back(ch);
        if (out.size() > 1024 * 256) return false;
    }
}

static void writeLine(int fd, const std::string &s) {
    auto writeFull = [&](const void *data, size_t len) -> bool {
        const char *p = (const char *)data;
        size_t left = len;
        while (left > 0) {
            ssize_t w = ::write(fd, p, left);
            if (w < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (w == 0) return false;
            p += (size_t)w;
            left -= (size_t)w;
        }
        return true;
    };

    (void)writeFull(s.data(), s.size());
    const char nl = '\n';
    (void)writeFull(&nl, 1);
}

static uint32_t parseU32(const json &j, const char *key, uint32_t fallback) {
    if (!j.contains(key)) return fallback;
    try {
        return j.at(key).get<uint32_t>();
    } catch (...) {
        return fallback;
    }
}

static std::string modeToStr(uint32_t mode) {
    if (mode == PLAN_CMD_MODE_EXACT) return "exact";
    if (mode == PLAN_CMD_MODE_NEXT_BARRIER) return "next_barrier";
    return "none";
}

static json cmdToJson(const PlanCommand &c) {
    return json{
        {"seq", c.seq},
        {"mode", modeToStr(c.mode)},
        {"stageIndex", c.stageIndex},
        {"triggerPos", c.triggerPos},
        {"triggerLayer", c.triggerLayer},
        {"fromNodeIndex", c.fromNodeIndex},
        {"toNodeIndex", c.toNodeIndex},
        {"cmdKind", c.cmdKind},
        {"nHeadsToMove", c.nHeadsToMove},
        {"nFfnToMove", c.nFfnToMove}
    };
}

static bool parseMode(const std::string &s, uint32_t &out) {
    if (s == "exact") { out = PLAN_CMD_MODE_EXACT; return true; }
    if (s == "next" || s == "next_barrier") { out = PLAN_CMD_MODE_NEXT_BARRIER; return true; }
    if (s == "none") { out = PLAN_CMD_MODE_NONE; return true; }
    return false;
}

std::unique_ptr<PlanUdsController> PlanUdsController::start(const std::string &socketPath, RootLlmInference *inference) {
#ifdef _WIN32
    (void)socketPath;
    (void)inference;
    return nullptr;
#else
    std::unique_ptr<PlanUdsController> ctrl(new PlanUdsController(socketPath, inference));

    // Launch background thread (C++11: no init-capture).
    PlanUdsController *c = ctrl.get();
    std::thread t([c]() { c->run(); });
    t.detach();

    return ctrl;
#endif
}

PlanUdsController::PlanUdsController(const std::string &socketPath, RootLlmInference *inference)
    : socketPath_(socketPath), inference_(inference) {}

PlanUdsController::~PlanUdsController() {
#ifndef _WIN32
    stop_ = true;
    closeServer();
#endif
}

void PlanUdsController::closeServer() {
#ifndef _WIN32
    if (serverFd_ >= 0) {
        ::close(serverFd_);
        serverFd_ = -1;
    }
    if (!socketPath_.empty()) {
        ::unlink(socketPath_.c_str());
    }
#endif
}

void PlanUdsController::run() {
#ifdef _WIN32
    return;
#else
    if (socketPath_.empty()) return;

    ::unlink(socketPath_.c_str());

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        std::perror("socket(AF_UNIX)");
        return;
    }

    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (socketPath_.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "[plan-uds] socket path too long: %s\n", socketPath_.c_str());
        ::close(fd);
        return;
    }
    std::strncpy(addr.sun_path, socketPath_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(fd, (sockaddr *)&addr, sizeof(addr)) != 0) {
        std::perror("bind(AF_UNIX)");
        ::close(fd);
        return;
    }

    if (::listen(fd, 16) != 0) {
        std::perror("listen(AF_UNIX)");
        ::close(fd);
        return;
    }

    serverFd_ = fd;
    std::fprintf(stderr, "[plan-uds] listening on %s\n", socketPath_.c_str());

    while (!stop_) {
        int cfd = ::accept(fd, nullptr, nullptr);
        if (cfd < 0) {
            if (stop_) break;
            continue;
        }

        std::string line;
        if (!readLine(cfd, line)) {
            ::close(cfd);
            continue;
        }

        json resp;
        try {
            json req = json::parse(line);
            const std::string op = req.value("op", "");

            if (op == "set_plan") {
                if (!req.contains("cmd")) throw std::runtime_error("missing cmd");
                const json &jcmd = req.at("cmd");

                PlanCommand cmd = makeEmptyPlanCommand();
                cmd.magic = DLLAMA_PLAN_CMD_MAGIC;
                cmd.version = DLLAMA_PLAN_CMD_VERSION;
                cmd.seq = parseU32(jcmd, "seq", 0u);

                const std::string modeStr = jcmd.value("mode", "");
                if (!parseMode(modeStr, cmd.mode)) {
                    throw std::runtime_error("bad mode (use exact/next_barrier)");
                }

                cmd.stageIndex = parseU32(jcmd, "stageIndex", 0u);
                cmd.fromNodeIndex = parseU32(jcmd, "fromNodeIndex", 0u);
                cmd.toNodeIndex = parseU32(jcmd, "toNodeIndex", 1u);
                cmd.cmdKind = parseU32(jcmd, "cmdKind", PLAN_CMD_KIND_BOTH);
                cmd.nHeadsToMove = parseU32(jcmd, "nHeadsToMove", 1u);
                cmd.nFfnToMove = parseU32(jcmd, "nFfnToMove", 256u);

                if (cmd.mode == PLAN_CMD_MODE_EXACT) {
                    if (!jcmd.contains("triggerPos") || !jcmd.contains("triggerLayer")) {
                        throw std::runtime_error("exact mode requires triggerPos + triggerLayer");
                    }
                    cmd.triggerPos = parseU32(jcmd, "triggerPos", 0xFFFFFFFFu);
                    cmd.triggerLayer = parseU32(jcmd, "triggerLayer", 0xFFFFFFFFu);
                } else {
                    cmd.triggerPos = 0xFFFFFFFFu;
                    cmd.triggerLayer = 0xFFFFFFFFu;
                }

                const uint64_t cacheSeq = planCommandCache().store(cmd);
                resp = json{{"ok", true}, {"cacheSeq", cacheSeq}, {"cmd", cmdToJson(cmd)}};
            } else if (op == "clear") {
                const uint64_t cacheSeq = planCommandCache().clear();
                resp = json{{"ok", true}, {"cacheSeq", cacheSeq}};
            } else if (op == "status") {
                const PlanCommandSnapshot snap = planCommandCache().load();
                resp = json{{"ok", true}, {"cacheSeq", snap.cacheSeq}, {"cmd", cmdToJson(snap.cmd)}};
                if (inference_ != nullptr) {
                    resp["position"] = inference_->getPosition();
                    resp["batchSize"] = inference_->getBatchSize();
                    resp["perfSamples"] = (uint32_t)inference_->getLastPerf().size();
                }
            } else if (op == "perf") {
                resp = json{{"ok", true}};
                json arr = json::array();
                if (inference_ != nullptr) {
                    for (const auto &p : inference_->getLastPerf()) {
                        arr.push_back(json{
                            {"position", p.position},
                            {"batchSize", p.batchSize},
                            {"nodeIndex", p.nodeIndex},
                            {"stageIndex", p.stageIndex},
                            {"execUs", p.execUs},
                            {"syncUs", p.syncUs}
                        });
                    }
                }
                resp["perf"] = arr;
            } else if (op == "ping") {
                resp = json{{"ok", true}};
            } else {
                resp = json{{"ok", false}, {"error", "unknown op"}};
            }
        } catch (const std::exception &e) {
            resp = json{{"ok", false}, {"error", e.what()}};
        } catch (...) {
            resp = json{{"ok", false}, {"error", "unknown error"}};
        }

        writeLine(cfd, resp.dump());
        ::close(cfd);
    }

    closeServer();
#endif
}
