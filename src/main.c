#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* x = create_ones_tensor((u32[]){2, 3}, 2, false);
    Tensor* labels = create_tensor((f32[]){0, 2}, (u32[]){2, 1}, 2, false);

    Tensor* a = create_ones_tensor((u32[]){3, 4}, 2, true);
    Tensor* w_1 = mat_mul(x, a);
    Tensor* b_1 = create_ones_tensor((u32[]){4}, 1, true);
    Tensor* hidden = mat_add(w_1, b_1);
    Tensor* logits = ReLU(hidden);

    Tensor* loss = cross_entropy_loss(logits, labels);

    for (u32 i = 0; i < a->size; i++) {
        a->grad[i] = 0.0f;
    }
    for (u32 i = 0; i < w_1->size; i++) {
        w_1->grad[i] = 0.0f;
    }
    for (u32 i = 0; i < b_1->size; i++) {
        b_1->grad[i] = 0.0f;
    }
    for (u32 i = 0; i < hidden->size; i++) {
        hidden->grad[i] = 0.0f;
    }
    for (u32 i = 0; i < logits->size; i++) {
        logits->grad[i] = 0.0f;
    }

    backward(loss);

    free_tensor(x);
    free_tensor(w_1);
    free_tensor(b_1);
    free_tensor(a);
    free_tensor(hidden);
    free_tensor(logits);
    free_tensor(labels);
    free_tensor(loss);

    return 0;
}