#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* logits = create_tensor((f32[]){2.0, 1.0, 0.1, 0.5, 2.5, 1.0}, (u32[]){2, 3}, 2, true);
    Tensor* labels = create_tensor((f32[]){0, 2}, (u32[]){2, 1}, 2, false);
    Tensor* tmp = Softmax(logits);
    Tensor* loss = cross_entropy_loss(logits, labels);
    backward(loss);

    printf("logits softmax: %f\n", tmp->data[0]);
    printf("logits softmax: %f\n", tmp->data[1]);
    printf("logits softmax: %f\n", tmp->data[2]);
    printf("logits softmax: %f\n", tmp->data[3]);
    printf("logits softmax: %f\n", tmp->data[4]);
    printf("logits softmax: %f\n", tmp->data[5]);

    free_tensor(logits);
    free_tensor(labels);
    free_tensor(loss);
    free_tensor(tmp);

    return 0;
}