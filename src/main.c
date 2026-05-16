#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* a = create_tensor((f32[]){1, 2, 3, 1, 3, 1, 1, -1, -1},(u32[]){3, 3}, 2, true);
    Tensor* b = create_tensor((f32[]){2, 2, 1, 3, 2, 2, 1, 3, 1}, (u32[]){3, 3}, 2, true);
    Tensor* c = mat_mul(a, b);
    
    Tensor* d = create_ones_tensor((u32[]){3, 3}, 2, true);
    Tensor* e = mat_add(c, d);

    Tensor* f = ReLU(e);

    for (u32 i = 0; i < f->size; i++) {
        f->grad[i] = 1.0f;
    }
    backward(f);

    printf("\nGradients:\n");
    printf("a grad: %f\n", a->grad[0]);
    printf("b grad: %f\n", b->grad[0]);
    printf("c grad: %f\n", c->grad[0]);
    printf("d grad: %f\n", d->grad[0]);
    printf("e grad: %f\n", e->grad[0]);
    printf("f grad: %f\n", f->grad[0]);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(d);
    free_tensor(e);
    free_tensor(f);

    return 0;
}