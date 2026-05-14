#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* a = create_tensor((f32[]){1, 2, 3, 1, 3, 1, 1, 1, 1},(u32[]){3, 3}, 2, true);
    Tensor* b = create_tensor((f32[]){2, 2, 1, 3, 2, 2, 1, 3, 1, 1, 1, 3}, (u32[]){3, 4}, 2, true);
    Tensor* c = mat_mul(a, b);

    print_tensor(a);
    print_tensor(b);
    print_tensor(c);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);

    return 0;
}