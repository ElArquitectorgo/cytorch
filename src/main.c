#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

u32 argmax(f32* values, u32 n) {
    u32 max_idx = 0;
    for (u32 i = 1; i < n; i++) {
        if (values[i] > values[max_idx]) {
             max_idx = i;
        }
    }
    return max_idx;
}

void test(Tensor** model) {
    Tensor* test = load_dataset("dataset/test_data_mnist.bin", 10000, 784);
    Tensor* test_labels = load_dataset("dataset/test_labels_mnist.bin", 10000, 1);

    Tensor* xw_1 = mat_mul(test, model[0]); // 10000 x 64
    Tensor* z_1 = mat_add(xw_1, model[1]); // 10000 x 64
    Tensor* h_1 = ReLU(z_1); // 10000 x 64

    Tensor* hw_2 = mat_mul(h_1, model[2]); // 10000 x 32
    Tensor* z_2 = mat_add(hw_2, model[3]); // 10000 x 32
    Tensor* h_2 = ReLU(z_2); // 10000 x 32

    Tensor* hw_3 = mat_mul(h_2, model[4]); // 10000 x 10
    Tensor* logits = mat_add(hw_3, model[5]); // 10000 x 10
    Tensor* predictions = Softmax(logits);

    u32 n_samples = test->shape[0];
    u32 correct = 0;
    for (u32 i = 0; i < n_samples; i++) {
        u32 pred = argmax(&predictions->data[i * 10], 10);
        u32 label = (u32)test_labels->data[i];
        if (pred == label) correct++;
    }
    
    printf("\nAccuracy: %f\n", (f32)correct / n_samples);

    free_tensor(xw_1);
    free_tensor(z_1);
    free_tensor(h_1);
    free_tensor(hw_2);
    free_tensor(z_2);
    free_tensor(h_2);
    free_tensor(hw_3);
    free_tensor(logits);
    free_tensor(predictions);

    free_tensor(test);
    free_tensor(test_labels);
}

void train(Tensor** model, Tensor* x, Tensor* labels, u32 batch_size, f32 learning_rate, f32 momentum) {
    for (u32 epoch = 0; epoch < 100; epoch++) {
        printf("\nEpoch %u\n", epoch + 1);

        u32 num_batches = x->shape[0] / batch_size;

        for (u32 batch = 0; batch < num_batches; batch++) {
            Tensor* X_train = create_zeros_tensor((u32[]){batch_size, 784}, 2, false);
            Tensor* y_train = create_zeros_tensor((u32[]){batch_size}, 1, false);
            for (u32 i = 0; i < batch_size; i++) {
                u32 row = batch * batch_size + i;
                memcpy(&X_train->data[i * 784], &x->data[row * 784], 784 * sizeof(f32));
                y_train->data[i] = labels->data[row];
            }

            Tensor* xw_1 = mat_mul(X_train, model[0]); // 64 x 64
            Tensor* z_1 = mat_add(xw_1, model[1]); // 64 x 64
            Tensor* h_1 = ReLU(z_1); // 64 x 64

            Tensor* hw_2 = mat_mul(h_1, model[2]); // 64 x 32
            Tensor* z_2 = mat_add(hw_2, model[3]); // 64 x 32
            Tensor* h_2 = ReLU(z_2); // 64 x 32

            Tensor* hw_3 = mat_mul(h_2, model[4]); // 64 x 10
            Tensor* logits = mat_add(hw_3, model[5]); // 64 x 10

            Tensor* loss = cross_entropy_loss(logits, y_train);
            backward(loss);

            for (u32 i = 0; i < 6; i++) {
                sgd_momentum_step(model[i], learning_rate, momentum);
                model[i]->visited = false;
            }

            printf("Batch %u/%u | Loss: %f\r", batch + 1, num_batches, loss->data[0]);
            fflush(stdout);

            free_tensor(xw_1);
            free_tensor(z_1);
            free_tensor(h_1);
            free_tensor(hw_2);
            free_tensor(z_2);
            free_tensor(h_2);
            free_tensor(hw_3);
            free_tensor(logits);
            free_tensor(loss);

            free_tensor(X_train);
            free_tensor(y_train);
        }
        test(model);
    }
}

int main() {
    srand(time(0));

    Tensor* x = load_dataset("dataset/train_data_mnist.bin", 60000, 784);
    Tensor* labels = load_dataset("dataset/train_labels_mnist.bin", 60000, 1);

    Tensor* w_1 = create_random_tensor((u32[]){784, 64}, 2, true);
    Tensor* b_1 = create_random_tensor((u32[]){64}, 1, true);

    Tensor* w_2 = create_random_tensor((u32[]){64, 32}, 2, true);
    Tensor* b_2 = create_random_tensor((u32[]){32}, 1, true);

    Tensor* w_3 = create_random_tensor((u32[]){32, 10}, 2, true);
    Tensor* b_3 = create_random_tensor((u32[]){10}, 1, true);

    Tensor** model = malloc(6 * sizeof(Tensor*));
    model[0] = w_1;
    model[1] = b_1;
    model[2] = w_2;
    model[3] = b_2;
    model[4] = w_3;
    model[5] = b_3;

    u32 batch_size = 64;
    f32 learning_rate = 0.001f;
    f32 momentum = 0.9f;

    train(model, x, labels, batch_size, learning_rate, momentum);
    
    printf("\n");

    free_tensor(x);
    free_tensor(labels);
    free_tensor(w_1);
    free_tensor(b_1);
    free_tensor(w_2);
    free_tensor(b_2);
    free_tensor(w_3);
    free_tensor(b_3);
    free(model);

    return 0;
}