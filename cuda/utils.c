#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void display_3dmat(char *name, float *M, int A, int B, int C) {
    // A x B x C
    printf("\nMatrix: %s\n", name);

    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < B; j++)
        {
            for (int k = 0; k < C; k++)
            {
                // skip batches, skip rows, skip columns
                int offset = i * B * C + j * C + k;
                printf("%0.4f ", M[offset]);
            }
            printf("\n");
        }
        printf("--\n");
    }
}

void display_2dmat(char *name, float *M, int A, int B) {
    // A x B
    printf("\nMatrix: %s\n", name);

    for (int i = 0; i < A; i++)
    {
        for (int j = 0; j < B; j++)
        {
            int offset = i * B + j;
            printf("%0.1f ", M[offset]);
        }
        printf("\n");
    }
}

void fill_random(float *X, size_t sz, int modu) {
    for (int i=0; i<sz; i++) {
        X[i] = rand() % modu;
    }
}