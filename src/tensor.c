#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

Tensor* create_1D_tensor(f32 data, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));
    t->data = malloc(sizeof(f32));
    *(t->data) = data; // is this safe?
    t->shape = malloc(sizeof(u32));
    t->shape[0] = 1;
    t->size = 1;
    t->num_dims = 1;
    t->grad = 0.0f;
    t->grad_op = None;
    t->requires_grad = requires_grad;
    t->is_leaf = true;
    t->visited = false;
    t->next_functions = NULL;
    t->num_next_functions = 0;
    return t;
}

Tensor* create_tensor(f32* data, u32* shape, u32 size, u32 num_dims, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));
    t->data = malloc(sizeof(f32) * size);
    t->data = data;

    t->shape = malloc(num_dims * sizeof(u32));
    for (u32 i = 0; i < num_dims; i++) {
        t->shape[i] = shape[i];
    }
    t->size = size;
    t->num_dims = num_dims;
    
    t->grad = 0.0f;
    t->grad_op = None;
    t->requires_grad = requires_grad;
    t->is_leaf = true;
    t->visited = false;

    t->next_functions = NULL;
    t->num_next_functions = 0;
    return t;
}

Tensor* create_zero_tensor(u32* shape, u32 num_dims, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));

    u32 num_values = 1;
    for (u32 i = 0; i < num_dims; i++) {
        num_values *= shape[i];
    }

    t->data = malloc(sizeof(f32) * num_values);
    for (u32 i = 0; i < num_values; i++) {
        t->data[i] = 0.0f;
    }

    t->shape = malloc(num_dims * sizeof(u32));
    for (u32 i = 0; i < num_dims; i++) {
        t->shape[i] = shape[i];
    }

    t->size = num_values;
    t->num_dims = num_dims;

    t->grad = 0.0f;
    
    t->grad_op = None;
    t->requires_grad = requires_grad;
    t->is_leaf = true;
    t->visited = false;

    t->next_functions = NULL;
    t->num_next_functions = 0;
    return t;
}

void print_tensor_recursive(Tensor* t, u8 dim, u32 offset) {
    printf("[");

    u32 stride = 1;
    for (u32 i = dim + 1; i < t->num_dims; i++) {
        stride *= t->shape[i];
    }

    for (u32 i = 0; i < t->shape[dim]; i++) {
        u32 current_offset = offset + i * stride;

        if (dim == t->num_dims - 1) {
            printf("%8.4f", t->data[current_offset]);
        } else {
            print_tensor_recursive(t, dim + 1, current_offset);
        }

        if (i < t->shape[dim] - 1) {
            if (dim == t->num_dims - 1) {
                printf(", ");
            } else if (dim == t->num_dims - 2) {
                printf(",\n");
                for (u8 k = 0; k <= 7 + dim; k++) printf(" ");
            } else {
                printf(",\n");
                for (u8 k = 0; k < t->num_dims - dim - 2; k++) printf("\n");
                for (u8 k = 0; k <= 7 + dim; k++) printf(" ");
            }
        }
    }

    printf("]");
}

void print_tensor(Tensor* t) {
    printf("tensor(");
    print_tensor_recursive(t, 0, 0);
    printf(")\n");
}

void free_tensor(Tensor* t) {
    if (t->next_functions != NULL) {
        free(t->next_functions);
    }
    if (t->data != NULL) {
        free(t->data);
        free(t->shape);
    }
    free(t);
}