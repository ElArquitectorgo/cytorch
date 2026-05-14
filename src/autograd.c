#include "autograd.h"

void backward(Tensor* t) {
    if (!t || t->visited) return;
    t->visited = true;

    if (t->grad_op == MUL) {
        mul_backward(t);
    }
    else if (t->grad_op == DIV) {
        div_backward(t);
    }
    else if (t->grad_op == ADD) {
        add_backward(t);
    }
    for (u8 i = 0; i < t->num_next_functions; i++) {
        backward(t->next_functions[i]);
    }
}

void add_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];
    
    if (a->requires_grad) {
        a->grad[0] += t->grad[0];
    }
    if (b->requires_grad) {
        b->grad[0] += t->grad[0];
    }
}

void mul_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    if (a->requires_grad) {
        a->grad[0] += t->grad[0] * b->data[0];
    }
    if (b->requires_grad) {
        b->grad[0] += t->grad[0] * a->data[0];
    }
}

void div_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    if (a->requires_grad) {
        a->grad[0] += t->grad[0] / b->data[0];
    }
    if (b->requires_grad) {
        b->grad[0] += -t->grad[0] * a->data[0] / (b->data[0] * b->data[0]);
    }
}