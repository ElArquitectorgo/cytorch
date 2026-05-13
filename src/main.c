#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    u32 shape[3] = {2, 3, 4};
    Tensor* a = create_zero_tensor(shape, 3, true);

    print_tensor(a);

    free_tensor(a);

    return 0;
}