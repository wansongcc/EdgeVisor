#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

// Container-local bounded client: no Python, Docker socket or host networking.
int main(int argc, char **argv) {
    int fd = -1;
    try {
        if (argc != 3) throw std::runtime_error("usage: plan_client socket-path JSON");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (std::strlen(argv[1]) >= sizeof(address.sun_path))
            throw std::runtime_error("socket path too long");
        std::strcpy(address.sun_path, argv[1]);
        fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) throw std::runtime_error("socket failed");
        timeval timeout{5, 0};
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)))
            throw std::runtime_error("socket timeout failed");
        if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)))
            throw std::runtime_error(std::string("connect: ") + std::strerror(errno));
        const std::string request = std::string(argv[2]) + "\n";
        size_t sent = 0;
        while (sent < request.size()) {
            const ssize_t n = send(fd, request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("send failed");
            sent += static_cast<size_t>(n);
        }
        std::string response;
        char c;
        while (response.size() < 1024u * 1024u) {
            const ssize_t n = recv(fd, &c, 1, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n != 1) throw std::runtime_error("incomplete response or timeout");
            response += c;
            if (c == '\n') break;
        }
        if (response.empty() || response.back() != '\n')
            throw std::runtime_error("response too large");
        std::cout << response;
        close(fd);
        return 0;
    } catch (const std::exception &e) {
        if (fd >= 0) close(fd);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
