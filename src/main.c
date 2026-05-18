#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* logits = create_tensor((f32[]){2.0, 1.0, 0.1}, (u32[]){1, 3}, 2, true);
    Tensor* labels = create_tensor((f32[]){0}, (u32[]){1, 1}, 2, false);

    Tensor* loss = cross_entropy_loss(logits, labels);
    backward(loss);

    printf("Loss: %f\n", loss->data[0]);
    printf("\nGradients:\n");
    printf("logits grad: %f\n", logits->grad[0]);
    printf("logits grad: %f\n", logits->grad[1]);
    printf("logits grad: %f\n", logits->grad[2]);

    free_tensor(logits);
    free_tensor(labels);
    free_tensor(loss);

    return 0;
}