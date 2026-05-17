#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

void backward(Tensor* t);

void mul_backward(Tensor* t);
void div_backward(Tensor* t);

void mat_add_backward(Tensor* t);
void mat_mul_backward(Tensor* t);

void relu_backward(Tensor* t);
void softmax_backward(Tensor* t);

void cross_entropy_loss_backward(Tensor* t, Tensor* labels);

#endif