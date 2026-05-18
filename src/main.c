#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* x = create_ones_tensor((u32[]){2, 3}, 2, false);
    Tensor* labels = create_tensor((f32[]){0, 2}, (u32[]){2, 1}, 2, false);

    Tensor* a = create_random_tensor((u32[]){3, 4}, 2, true);
    Tensor* w_1 = mat_mul(x, a);
    Tensor* b_1 = create_random_tensor((u32[]){4}, 1, true);
    Tensor* hidden = mat_add(w_1, b_1);
    Tensor* logits = ReLU(hidden);

    Tensor* loss = cross_entropy_loss(logits, labels);

    Tensor** model = malloc(4 * sizeof(Tensor*));
    model[0] = a;
    model[1] = b_1;
    model[2] = hidden;
    model[3] = logits;

    for (u32 i = 0; i < 4; i++) {
        for (u32 j = 0; j < model[i]->size; j++) {
            model[i]->grad[j] = 0.0f;
        }
    }

    backward(loss);
    f32 learning_rate = 0.01f;
    for (u32 i = 0; i < 4; i++) {
        for (u32 j = 0; j < model[i]->size; j++) {
            model[i]->data[j] -= learning_rate * model[i]->grad[j];
            model[i]->grad[j] = 0.0f;
        }
    }
    print_tensor(a);

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