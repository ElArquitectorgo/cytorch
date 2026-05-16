#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "ops.h"
#include "autograd.h"


int main() {
    Tensor* a = create_tensor((f32[]){1, 2, 3, 1, 3, 1, 1, 1, 1},(u32[]){3, 3}, 2, true);
    Tensor* b = create_tensor((f32[]){2, 2, 1, 3, 2, 2, 1, 3, 1, 1, 1, 3}, (u32[]){3, 4}, 2, true);
    Tensor* c = mat_mul(a, b);
    
    Tensor* d = create_tensor((f32[]){2, 1, 3, 1}, (u32[]){2, 2}, 2, true);
    Tensor* e = create_tensor((f32[]){2, 2, 2, 2}, (u32[]){2, 2}, 2, true);
    Tensor* f = mat_add(d, e);

    print_tensor(a);
    print_tensor(b);
    printf("\n");
    print_tensor(c);
    printf("\n");
    printf("\n");
    print_tensor(d);
    print_tensor(e);
    printf("\n");
    print_tensor(f);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(d);
    free_tensor(e);
    free_tensor(f);

    return 0;
}