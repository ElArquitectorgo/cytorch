#include "tensor.h"
#include <stdlib.h>

Tensor* create_tensor(f32 data, bool requires_grad) {
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

void free_tensor(Tensor* t) {
    if (t->next_functions != NULL) {
        free(t->next_functions);
    }
    free(t);
}