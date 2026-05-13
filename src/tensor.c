#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

Tensor* create_tensor(Array* data, bool requires_grad) {
    Tensor* t = malloc(sizeof(Tensor));

    t->data = data;
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

    t->data = malloc(sizeof(Array));
    t->data->values = malloc(num_values * sizeof(f32));
    for (u32 i = 0; i < num_values; i++) {
        t->data->values[i] = 0.0f;
    }

    t->data->shape = malloc(num_dims * sizeof(u32));
    for (u32 i = 0; i < num_dims; i++) {
        t->data->shape[i] = shape[i];
    }

    t->data->size = num_values;
    t->data->num_dims = num_dims;

    t->grad = 0.0f;
    
    t->grad_op = None;
    t->requires_grad = requires_grad;
    t->is_leaf = true;
    t->visited = false;

    t->next_functions = NULL;
    t->num_next_functions = 0;
    return t;
}

void print_tensor_recursive(Array* arr, u8 dim, u32 offset) {
    printf("[");

    u32 stride = 1;
    for (u32 i = dim + 1; i < arr->num_dims; i++) {
        stride *= arr->shape[i];
    }

    for (u32 i = 0; i < arr->shape[dim]; i++) {
        u32 current_offset = offset + i * stride;

        if (dim == arr->num_dims - 1) {
            printf("%8.4f", arr->values[current_offset]);
        } else {
            print_tensor_recursive(arr, dim + 1, current_offset);
        }

        if (i < arr->shape[dim] - 1) {
            if (dim == arr->num_dims - 1) {
                printf(", ");
            } else if (dim == arr->num_dims - 2) {
                printf(",\n");
                for (u8 k = 0; k <= 7 + dim; k++) printf(" ");
            } else {
                printf(",\n");
                for (u8 k = 0; k < arr->num_dims - dim - 2; k++) printf("\n");
                for (u8 k = 0; k <= 7 + dim; k++) printf(" ");
            }
        }
    }

    printf("]");
}

void print_tensor(Tensor* t) {
    printf("tensor(");
    print_tensor_recursive(t->data, 0, 0);
    printf(")\n");
}

void free_tensor(Tensor* t) {
    if (t->next_functions != NULL) {
        free(t->next_functions);
    }
    if (t->data != NULL) {
        free(t->data->values);
        free(t->data->shape);
        free(t->data);
    }
    free(t);
}