#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include "base.h"

typedef enum { 
    MUL,
    DIV,
    ADD,
    None,
} op_type;

typedef struct Tensor {
    f32 data;
    f32 grad;
    op_type grad_op;
    struct Tensor** next_functions;
    u8 num_next_functions;
    bool requires_grad;
    bool is_leaf;
    bool visited;
} Tensor;

Tensor* create_tensor(f32 data, bool requires_grad);
void free_tensor(Tensor* t);

#endif