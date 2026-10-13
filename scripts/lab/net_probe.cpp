#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char *what) {
    if (!ok) throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}
int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: net_probe serve|send IPv4 port bytes");
        const bool serve = std::string(argv[1]) == "serve";
        if (!serve && std::string(argv[1]) != "send") throw std::runtime_error("bad mode");
        const int port = std::stoi(argv[3]);
        const size_t bytes = std::stoull(argv[4]);
        if (port < 1024 || port > 65535 || bytes == 0 || bytes > 64u*1024u*1024u)
            throw std::runtime_error("bad port or byte count");
        int fd = socket(AF_INET, SOCK_STREAM, 0); check(fd >= 0, "socket");
        timeval timeout{15, 0};
        check(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0, "recv timeout");
        check(setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0, "send timeout");
        sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)port);
        check(inet_pton(AF_INET, argv[2], &addr.sin_addr) == 1, "IPv4");
        if (serve) {
            int one = 1;
            check(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0, "reuse");
            check(bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0, "bind");
            check(listen(fd, 1) == 0, "listen");
            std::cerr << "READY\n" << std::flush;
            int peer = accept(fd, nullptr, nullptr); check(peer >= 0, "accept");
            close(fd); fd = peer;
            check(setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0, "peer timeout");
        } else {
            check(connect(fd, (sockaddr*)&addr, sizeof(addr)) == 0, "connect");
        }
        std::vector<unsigned char> buffer(65536u, 0x5a);
        size_t done = 0;
        const auto start = std::chrono::steady_clock::now();
        while (done < bytes) {
            const size_t want = std::min(buffer.size(), bytes-done);
            ssize_t got = serve ? recv(fd, buffer.data(), want, 0) : send(fd, buffer.data(), want, MSG_NOSIGNAL);
            check(got > 0, serve ? "recv" : "send");
            if (serve) for (ssize_t i = 0; i < got; ++i)
                if (buffer[(size_t)i] != 0x5a) throw std::runtime_error("payload mismatch");
            done += (size_t)got;
        }
        if (serve) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const std::string result = "{\"bytes\":"+std::to_string(done)+",\"seconds\":"+std::to_string(seconds)+
                ",\"mbps\":"+std::to_string(done*8.0/seconds/1000000.0)+"}\n";
            size_t sent = 0;
            while (sent < result.size()) {
                ssize_t n = send(fd, result.data()+sent, result.size()-sent, MSG_NOSIGNAL);
                check(n > 0, "ack"); sent += (size_t)n;
            }
            std::cout << result;
        } else {
            // Wait for receiver completion, not just a successful buffered send.
            std::string ack;
            char byte;
            while (ack.size() < 1024u) {
                check(recv(fd, &byte, 1u, 0) == 1, "receiver ack");
                ack += byte; if (byte == '\n') break;
            }
            if (ack.empty() || ack.back() != '\n') throw std::runtime_error("incomplete ack");
            std::cout << ack;
        }
        close(fd); return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
