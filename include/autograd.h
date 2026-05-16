#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

void backward(Tensor* t);

void mul_backward(Tensor* t);
void div_backward(Tensor* t);

void mat_add_backward(Tensor* t);
void mat_mul_backward(Tensor* t);

void ReLU_backward(Tensor* t);

#endif