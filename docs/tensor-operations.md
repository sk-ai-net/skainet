# Tensor Operations in SK-AI-Net

This document provides detailed information about the tensor operations available in SK-AI-Net, including their behavior, expected inputs, outputs, and examples of usage.

## Table of Contents

- [Basic Operations](#basic-operations)
  - [Addition](#addition)
  - [Multiplication](#multiplication)
  - [Matrix Multiplication](#matrix-multiplication)
  - [Power](#power)
- [Activation Functions](#activation-functions)
  - [ReLU](#relu)
- [Shape Operations](#shape-operations)
  - [Reshaping](#reshaping)
  - [Slicing](#slicing)
- [Advanced Operations](#advanced-operations)
  - [Broadcasting](#broadcasting)
  - [Reduction Operations](#reduction-operations)

## Basic Operations

### Addition

The addition operation adds two tensors element-wise.

**Signature:**
```kotlin
operator fun Tensor.plus(other: Tensor): Tensor
```

**Behavior:**
- If the tensors have the same shape, each element in the result is the sum of the corresponding elements in the input tensors.
- If the tensors have different shapes, broadcasting rules are applied if possible.

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0))
val b = DoublesTensor(Shape(2, 2), doubleArrayOf(5.0, 6.0, 7.0, 8.0))
val c = a + b // Results in [[6.0, 8.0], [10.0, 12.0]]
```

### Multiplication

The multiplication operation multiplies two tensors element-wise.

**Signature:**
```kotlin
operator fun Tensor.times(other: Tensor): Tensor
```

**Behavior:**
- If the tensors have the same shape, each element in the result is the product of the corresponding elements in the input tensors.
- If the tensors have different shapes, broadcasting rules are applied if possible.

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0))
val b = DoublesTensor(Shape(2, 2), doubleArrayOf(5.0, 6.0, 7.0, 8.0))
val c = a * b // Results in [[5.0, 12.0], [21.0, 32.0]]
```

### Matrix Multiplication

The matrix multiplication operation performs the mathematical matrix product of two tensors.

**Signature:**
```kotlin
infix fun Tensor.matmul(other: Tensor): Tensor
```

**Behavior:**
- For 2D tensors, this performs standard matrix multiplication.
- The number of columns in the first tensor must match the number of rows in the second tensor.
- The result has shape (n, p) where n is the number of rows in the first tensor and p is the number of columns in the second tensor.

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
val b = DoublesTensor(Shape(3, 2), doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0))
val c = a matmul b // Results in [[58.0, 64.0], [139.0, 154.0]]
```

### Power

The power operation raises each element of a tensor to a specified power.

**Signature:**
```kotlin
fun DoublesTensor.pow(scalar: Double): TypedTensor<Double>
```

**Behavior:**
- Each element in the result is the corresponding element in the input tensor raised to the power of the scalar.

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0))
val b = a.pow(2.0) // Results in [[1.0, 4.0], [9.0, 16.0]]
```

## Activation Functions

### ReLU

The ReLU (Rectified Linear Unit) activation function replaces negative values with zero.

**Signature:**
```kotlin
fun relu(tensor: Tensor): Tensor
```

**Behavior:**
- Each element in the result is max(0, x) where x is the corresponding element in the input tensor.

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 2), doubleArrayOf(-1.0, 2.0, -3.0, 4.0))
val b = relu(a) // Results in [[0.0, 2.0], [0.0, 4.0]]
```

## Shape Operations

### Reshaping

Reshaping changes the shape of a tensor without changing its data.

**Signature:**
```kotlin
fun Tensor.reshape(newShape: Shape): Tensor
```

**Behavior:**
- The total number of elements in the new shape must match the total number of elements in the original shape.
- The data is rearranged to fit the new shape.

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
val b = a.reshape(Shape(3, 2)) // Results in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
```

### Slicing

Slicing extracts a subset of a tensor.

**Signature:**
```kotlin
operator fun Tensor.get(vararg slices: Slice): Tensor
```

**Behavior:**
- Each slice specifies a range of indices to include in the corresponding dimension.
- The result is a new tensor containing only the specified elements.

**Example:**
```kotlin
val a = DoublesTensor(Shape(3, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
val b = a[Slice(0, 2), Slice(1, 3)] // Results in [[2.0, 3.0], [5.0, 6.0]]
```

## Advanced Operations

### Broadcasting

Broadcasting allows operations between tensors of different shapes by implicitly expanding smaller tensors to match the shape of larger tensors.

**Rules:**
1. If the tensors have different ranks, prepend dimensions of size 1 to the smaller tensor until both tensors have the same rank.
2. For each dimension, if one tensor has size 1 in that dimension, it is expanded to match the size of the other tensor in that dimension.
3. If the sizes in any dimension don't match and neither is 1, an error is raised.

**Example:**
```kotlin
val a = DoublesTensor(Shape(3, 1), doubleArrayOf(1.0, 2.0, 3.0))
val b = DoublesTensor(Shape(1, 4), doubleArrayOf(10.0, 20.0, 30.0, 40.0))
val c = a + b // Results in a 3x4 tensor through broadcasting
```

### Reduction Operations

Reduction operations reduce a tensor along specified dimensions, producing a tensor with fewer dimensions.

**Common reduction operations:**
- Sum: Computes the sum of elements along specified dimensions
- Mean: Computes the mean of elements along specified dimensions
- Max/Min: Finds the maximum/minimum values along specified dimensions

**Example:**
```kotlin
val a = DoublesTensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
val sum = a.sum(dim = 1) // Results in [6.0, 15.0]
val mean = a.mean(dim = 0) // Results in [2.5, 3.5, 4.5]
```