#include <stdio.h>
#include <stdlib.h>
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
Tensor* mul(Tensor* a, Tensor* b);
Tensor* divide(Tensor* a, Tensor* b);
Tensor* add(Tensor* a, Tensor* b);
void free_tensor(Tensor* t);

void mul_backward(Tensor* t);
void div_backward(Tensor* t);
void add_backward(Tensor* t);
void backward(Tensor* t);

int main() {    
    Tensor* a = create_tensor(2, false);
    Tensor* b = create_tensor(2, false);

    Tensor* c = mul(a, b);
    c->requires_grad = true;
    Tensor* d = create_tensor(2, false);

    Tensor* e = mul(c, d);
    Tensor* f = create_tensor(2, false);

    Tensor* g = mul(e, f);
    Tensor* h = create_tensor(2, true);

    Tensor* i = divide(g, h);

    Tensor* j = add(i, h);

    Tensor* k = mul(j, i);
    k->grad = 1.0f; // Set the gradient of the output tensor to 1 for backpropagation
    backward(k);

    printf("a value: %f, grad: %f\n", a->data, a->grad);
    printf("b value: %f, grad: %f\n", b->data, b->grad);
    printf("c value: %f, grad: %f\n", c->data, c->grad);
    printf("d value: %f, grad: %f\n", d->data, d->grad);
    printf("e value: %f, grad: %f\n", e->data, e->grad);
    printf("f value: %f, grad: %f\n", f->data, f->grad);
    printf("g value: %f, grad: %f\n", g->data, g->grad);
    printf("h value: %f, grad: %f\n", h->data, h->grad);
    printf("i value: %f, grad: %f\n", i->data, i->grad);
    printf("j value: %f, grad: %f\n", j->data, j->grad);
    printf("k value: %f, grad: %f\n", k->data, k->grad);

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    free_tensor(d);
    free_tensor(e);
    free_tensor(f);
    free_tensor(g);
    free_tensor(h);
    free_tensor(i);
    free_tensor(j);
    free_tensor(k);
}

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

void mul_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    if (a->requires_grad) {
        a->grad += t->grad * b->data;
    }
    if (b->requires_grad) {
        b->grad += t->grad * a->data;
    }
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

void div_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];

    if (a->requires_grad) {
        a->grad += t->grad / b->data;
    }
    if (b->requires_grad) {
        b->grad += -t->grad * a->data / (b->data * b->data);
    }
}

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

void add_backward(Tensor* t) {
    Tensor* a = t->next_functions[0];
    Tensor* b = t->next_functions[1];
    
    if (a->requires_grad) {
        a->grad += t->grad;
    }
    if (b->requires_grad) {
        b->grad += t->grad;
    }
}

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