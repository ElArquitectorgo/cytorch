#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"
#include "optimizers.h"

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
    for (u32 epoch = 0; epoch < 10; epoch++) {
        printf("\nEpoch %u\n", epoch + 1);

        u32 num_batches = (x->shape[0]) / batch_size;

        for (u32 batch = 0; batch < num_batches; batch++) {
            Tensor* X_train = create_zeros_tensor((u32[]){batch_size, 1}, 2, false);
            Tensor* y_train = create_zeros_tensor((u32[]){batch_size, 1}, 2, false);
            for (u32 i = 0; i < batch_size; i++) {
                X_train->data[i] = x->data[(batch * batch_size + i) % x->shape[0]];
                y_train->data[i] = labels->data[(batch * batch_size + i) % labels->shape[0]];
            }

            

            Tensor* a = mat_mul(X_train, w_1);
            Tensor* logits = mat_add(a, b_1);
            //Tensor* logits = ReLU(hidden);

            Tensor* loss = cross_entropy_loss(logits, y_train);
            backward(loss);

            for (u32 i = 0; i < 2; i++) {
                sgd_momentum_step(model[i], learning_rate, momentum);
            }
            printf("\rLoss: %f\r", loss->data[0]);
            fflush(stdout);
        }
    }

    free_tensor(x);
    free_tensor(labels);
    free_tensor(w_1);
    free_tensor(b_1);

    return 0;
}