#ifndef UTILS_H
#define UTISL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

void display_3dmat(char *name, float *M, int A, int B, int C);
void display_2dmat(char *name, float *M, int A, int B);
void fill_random(float *X, size_t sz, int modu);

#ifdef __cplusplus
}
#endif
#endif
