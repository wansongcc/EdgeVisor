#ifndef PRODUCT_LOG_HPP
#define PRODUCT_LOG_HPP

// 0: device, topology, progress, generated text, final speed.
// 1: --verbose. 2: --debug.
void productSetLogLevel(int level);
int productLogLevel();

#endif
