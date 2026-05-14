#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef float f32;

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ABS(n) ((n) < 0 ? -(n) : (n))

#define ARRAY_SIZE(x) (sizeof(x)/sizeof((x)[0]))