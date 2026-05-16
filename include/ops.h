#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor* add(Tensor* a, Tensor* b);
Tensor* mul(Tensor* a, Tensor* b);
Tensor* divide(Tensor* a, Tensor* b);

Tensor* mat_mul(Tensor* a, Tensor* b);
Tensor* mat_add(Tensor* a, Tensor* b);

Tensor* ReLU(Tensor* a);

#endif