#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include "tensor.h"

void sgd_step(Tensor* t, f32 learning_rate);
void sgd_momentum_step(Tensor* t, f32 learning_rate, f32 momentum);

#endif