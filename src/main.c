#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"
#include "optimizers.h"


Tensor* load_dataset(const char* file_name, u32 rows, u32 cols) {
    Tensor* x = create_zeros_tensor((u32[]){rows, cols}, 2, false);

    FILE* f = fopen(file_name, "rb");

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    fread(x->data, 1, size, f);

    fclose(f);

    return x;
}

int main() {
    Tensor* x = load_dataset("dataset/train_data_mnist.bin", 60000, 784);
    Tensor* labels = load_dataset("dataset/train_labels_mnist.bin", 60000, 1);

    Tensor* w_1 = create_random_tensor((u32[]){784, 64}, 2, true);
    Tensor* b_1 = create_random_tensor((u32[]){64}, 1, true);

    Tensor* w_2 = create_random_tensor((u32[]){64, 32}, 2, true);
    Tensor* b_2 = create_random_tensor((u32[]){32}, 1, true);

    Tensor** model = malloc(4 * sizeof(Tensor*));
    model[0] = w_1;
    model[1] = b_1;
    model[2] = w_2;
    model[3] = b_2;

    u32 batch_size = 64;
    f32 learning_rate = 0.01f;
    f32 momentum = 0.9f;

    for (u32 epoch = 0; epoch < 100; epoch++) {
        printf("\nEpoch %u\n", epoch + 1);

        u32 num_batches = x->shape[0] / batch_size;

        for (u32 batch = 0; batch < num_batches; batch++) {
            Tensor* X_train = create_zeros_tensor((u32[]){batch_size, 784}, 2, false);
            Tensor* y_train = create_zeros_tensor((u32[]){batch_size, 1}, 2, false);
            for (u32 i = 0; i < batch_size; i++) {
                X_train->data[i] = x->data[(batch * batch_size + i) % x->shape[0]];
                y_train->data[i] = labels->data[(batch * batch_size + i) % labels->shape[0]];
            }

            Tensor* xw_1 = mat_mul(X_train, w_1);
            Tensor* z_1 = mat_add(xw_1, b_1);
            Tensor* h_1 = ReLU(z_1);
            Tensor* hw_2 = mat_mul(h_1, w_2);
            Tensor* logits = mat_add(hw_2, b_2);

            Tensor* loss = cross_entropy_loss(logits, y_train);
            backward(loss);

            for (u32 i = 0; i < 4; i++) {
                sgd_momentum_step(model[i], learning_rate, momentum);
            }

            printf("Batch %u/%u | Loss: %f\r", batch + 1, num_batches, loss->data[0]);
            fflush(stdout);

            free_tensor(xw_1);
            free_tensor(z_1);
            free_tensor(h_1);
            free_tensor(hw_2);
            free_tensor(logits);
            free_tensor(loss);

            free_tensor(X_train);
            free_tensor(y_train);
        }
    }
    printf("\n");

    free_tensor(x);
    free_tensor(labels);
    free_tensor(w_1);
    free_tensor(b_1);
    free_tensor(w_2);
    free_tensor(b_2);
    free(model);

    return 0;
}