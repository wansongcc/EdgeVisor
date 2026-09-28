#include "product_log.hpp"

static int g_productLogLevel = 0;

void productSetLogLevel(int level) {
    if (level < 0) level = 0;
    if (level > 2) level = 2;
    g_productLogLevel = level;
}

int productLogLevel() {
    return g_productLogLevel;
}
