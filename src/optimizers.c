#include "optimizers.h"

void sgd_step(Tensor* t, f32 learning_rate) {
    for (u32 i = 0; i < t->size; i++) {
        t->data[i] -= learning_rate * t->grad[i];
        t->grad[i] = 0.0f;
    }
}

void sgd_momentum_step(Tensor* t, f32 learning_rate, f32 momentum) {
    for (u32 i = 0; i < t->size; i++) {
        t->velocity[i] = momentum * t->velocity[i] - learning_rate * t->grad[i];
        t->data[i] += t->velocity[i];
        t->grad[i] = 0.0f;
    }
}