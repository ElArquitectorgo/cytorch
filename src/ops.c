#include "ops.h"
#include <stdlib.h>

Tensor* add(Tensor* a, Tensor* b) {
    Tensor* t = create_tensor(a->data + b->data, a->requires_grad || b->requires_grad);
    t->is_leaf = false;
    
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = ADD;
    }
    
    return t;
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* t = create_tensor(a->data * b->data, a->requires_grad || b->requires_grad);
    t->is_leaf = false;
    
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = MUL;
    }
    return t;
}

Tensor* divide(Tensor* a, Tensor* b) {
    Tensor* t = create_tensor(a->data / b->data, a->requires_grad || b->requires_grad);
    t->is_leaf = false;
    
    t->next_functions = malloc(2 * sizeof(Tensor*));
    t->num_next_functions = 2;
    t->next_functions[0] = a;
    t->next_functions[1] = b;

    if (t->requires_grad) {
        t->grad_op = DIV;
    }
    return t;
}