#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include "base.h"

typedef enum { 
    MUL,
    MAT_MUL,
    DIV,
    MAT_ADD,
    RELU,
    SOFTMAX,
    CEL,
    None,
} op_type;

typedef struct Tensor {
    f32* data;
    u32* shape;
    u32 size;
    u8 num_dims;
    f32* grad;
    op_type grad_op;
    f32* velocity;
    struct Tensor* labels;
    struct Tensor** next_functions;
    u8 num_next_functions;
    bool requires_grad;
    bool is_leaf;
    bool visited;
} Tensor;

Tensor* create_1D_tensor(f32 data, bool requires_grad);
Tensor* create_tensor(f32* data, u32* shape, u32 num_dims, bool requires_grad);
Tensor* create_random_tensor(u32* shape, u32 num_dims, bool requires_grad);
Tensor* create_zeros_tensor(u32* shape, u32 num_dims, bool requires_grad);
Tensor* create_ones_tensor(u32* shape, u32 num_dims, bool requires_grad);
void print_tensor(Tensor* t);
void free_tensor(Tensor* t);

#endif