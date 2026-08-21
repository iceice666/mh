/* mh host stdlib build tool.
 *
 * Compiled for the build host by build.rs and run once to generate the
 * serialized stdlib ROM table (mh_stdlib.h) and atom header
 * (mquickjs_atom.h). It is the default mqjs stdlib (mqjs_stdlib.c) plus
 * the single CONFIG_MH closure entry declared there.
 */
#define CONFIG_MH 1
#include "../vendor/mquickjs/mqjs_stdlib.c"
