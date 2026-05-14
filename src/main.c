#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* a = create_zero_tensor((u32[]){2, 3, 4}, 3, true);

    Tensor* b = create_2D_tensor((f32[]){1, 2, 3, 4, 5, 6, 7, 8, 9}, (u32[]){3, 3}, true);

    print_tensor(a);
    print_tensor(b);

    free_tensor(a);
    free_tensor(b);

    return 0;
}