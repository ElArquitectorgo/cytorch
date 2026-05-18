#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"

void sgd_momentum_step(Tensor* t, f32 learning_rate, f32 momentum) {
    for (u32 i = 0; i < t->size; i++) {
        t->velocity[i] = momentum * t->velocity[i] - learning_rate * t->grad[i];
        t->data[i] += t->velocity[i];
        t->grad[i] = 0.0f;
    }
}

int main() {
    Tensor* x = create_tensor((f32[]){1, 1, 1, 1, 0, 0, 0, 0},(u32[]){8, 1}, 2, false);
    Tensor* labels = create_tensor((f32[]){0, 0, 0, 0, 1, 1, 1, 1}, (u32[]){8, 1}, 2, false);

    Tensor* w_1 = create_random_tensor((u32[]){1, 4}, 2, true);
    Tensor* b_1 = create_random_tensor((u32[]){4}, 1, true);

    Tensor** model = malloc(2 * sizeof(Tensor*));
    model[0] = w_1;
    model[1] = b_1;

    u32 batch_size = 2;
    f32 learning_rate = 0.01f;
    f32 momentum = 0.9f;
    for (u32 epoch = 0; epoch < 500; epoch++) {
        Tensor* X_train = create_zeros_tensor((u32[]){batch_size, 1}, 2, false);
        Tensor* y_train = create_zeros_tensor((u32[]){batch_size, 1}, 2, false);
        for (u32 i = 0; i < batch_size; i++) {
            X_train->data[i] = x->data[(epoch * batch_size + i) % x->size];
            y_train->data[i] = labels->data[(epoch * batch_size + i) % labels->size];
        }

        printf("Epoch %u\n", epoch + 1);
        Tensor* a = mat_mul(X_train, w_1);
        Tensor* logits = mat_add(a, b_1);
        //Tensor* logits = ReLU(hidden);

        Tensor* loss = cross_entropy_loss(logits, y_train);
        backward(loss);
        
        for (u32 i = 0; i < 2; i++) {
            sgd_momentum_step(model[i], learning_rate, momentum);
        }
        printf("Loss: %f\n", loss->data[0]);
    }

    free_tensor(x);
    free_tensor(labels);
    free_tensor(w_1);
    free_tensor(b_1);

    return 0;
}