from pathlib import Path

p = Path("/home/cc/yhbian/B01_Copy_API/EdgeVisor/src/plan-controller.cpp")
s = p.read_text()

old = """PlanUdsController::~PlanUdsController() {
#ifndef _WIN32
    stop_.store(true);
    closeServer();
    if (worker_.joinable()) {
        worker_.join();
    }
#endif
}
"""
new = """PlanUdsController::~PlanUdsController() {
#ifndef _WIN32
    stop_.store(true);
    if (!socketPath_.empty()) {
        int wakeFd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (wakeFd >= 0) {
            sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socketPath_.c_str());
            (void)::connect(wakeFd, (sockaddr *)&addr, sizeof(addr));
            ::close(wakeFd);
        }
    }
    closeServer();
    if (worker_.joinable()) {
        worker_.join();
    }
#endif
}
"""
if new not in s:
    if old not in s:
        raise RuntimeError("PlanUdsController destructor pattern not found")
    s = s.replace(old, new, 1)

old = """    while (!stop_.load()) {
        int cfd = ::accept(fd, nullptr, nullptr);
        if (cfd < 0) {
            if (stop_.load()) break;
            continue;
        }

        std::string line;
"""
new = """    while (!stop_.load()) {
        int cfd = ::accept(fd, nullptr, nullptr);
        if (cfd < 0) {
            if (stop_.load()) break;
            continue;
        }
        if (stop_.load()) {
            ::close(cfd);
            break;
        }

        std::string line;
"""
if new not in s:
    if old not in s:
        raise RuntimeError("PlanUdsController accept loop pattern not found")
    s = s.replace(old, new, 1)

p.write_text(s)
print("Plan UDS destructor now wakes accept before join")
