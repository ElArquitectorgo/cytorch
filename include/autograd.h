#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

void backward(Tensor* t);

void mul_backward(Tensor* t);
void add_backward(Tensor* t);
void div_backward(Tensor* t);

#endif