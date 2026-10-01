#include <stdio.h>
#include <os/proc.h>

// lc0's main(), renamed by patches/lc0-ios.patch (built with -Dios_library=true).
// Runs synchronously; output goes to stdout/stderr.
int lc0_main(int argc, const char **argv);

// Stockfish's main(), renamed by scripts/build_stockfish.sh. Runs a single UCI
// command line (e.g. "bench ..." or "speedtest ...") and returns.
int stockfish_main_c(int argc, char **argv);
