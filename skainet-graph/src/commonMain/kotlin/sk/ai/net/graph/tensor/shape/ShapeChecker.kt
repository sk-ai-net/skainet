package sk.ai.net.graph.tensor.shape

import sk.ai.net.graph.tensor.Tensor

/**
 * A utility class for checking tensor shapes before operations.
 *
 * This class provides methods for validating tensor shapes before operations
 * are performed, and throws descriptive exceptions when shape mismatches occur.
 */
object ShapeChecker {
    /**
     * Checks if two tensors have the same shape.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @param operationName The name of the operation being performed.
     * @throws IllegalArgumentException If the shapes don't match.
     */
    fun checkSameShape(left: Tensor, right: Tensor, operationName: String) {
        if (left.shape != right.shape) {
            throw IllegalArgumentException(
                "Shape mismatch for $operationName operation: " +
                "left tensor has shape ${left.shape}, " +
                "right tensor has shape ${right.shape}"
            )
        }
    }

    /**
     * Checks if two tensors are compatible for matrix multiplication.
     *
     * For matrix multiplication, the number of columns in the left tensor
     * must equal the number of rows in the right tensor.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @throws IllegalArgumentException If the tensors are not compatible.
     */
    fun checkMatmulCompatible(left: Tensor, right: Tensor) {
        // Check that both tensors are at least 2D
        if (left.shape.rank < 2 || right.shape.rank < 2) {
            throw IllegalArgumentException(
                "Matrix multiplication requires at least 2D tensors: " +
                "left tensor has ${left.shape.rank} dimensions, " +
                "right tensor has ${right.shape.rank} dimensions"
            )
        }

        // Check that the inner dimensions match
        val leftCols = left.shape[1]
        val rightRows = right.shape[0]
        if (leftCols != rightRows) {
            throw IllegalArgumentException(
                "Incompatible shapes for matrix multiplication: " +
                "left tensor has shape ${left.shape}, " +
                "right tensor has shape ${right.shape}, " +
                "inner dimensions $leftCols and $rightRows must match"
            )
        }
    }

    /**
     * Checks if a tensor has the expected shape.
     *
     * @param tensor The tensor to check.
     * @param expectedShape The expected shape.
     * @param operationName The name of the operation being performed.
     * @throws IllegalArgumentException If the shape doesn't match the expected shape.
     */
    fun checkExpectedShape(tensor: Tensor, expectedShape: Shape, operationName: String) {
        if (tensor.shape != expectedShape) {
            throw IllegalArgumentException(
                "Shape mismatch for $operationName operation: " +
                "tensor has shape ${tensor.shape}, " +
                "expected shape $expectedShape"
            )
        }
    }

    /**
     * Checks if a tensor has the expected shape.
     * This overload accepts a List<Int> for backward compatibility.
     *
     * @param tensor The tensor to check.
     * @param expectedDimensions The expected dimensions as a list.
     * @param operationName The name of the operation being performed.
     * @throws IllegalArgumentException If the shape doesn't match the expected shape.
     */
    fun checkExpectedShape(tensor: Tensor, expectedDimensions: List<Int>, operationName: String) {
        checkExpectedShape(tensor, Shape(expectedDimensions.toIntArray()), operationName)
    }

    /**
     * Checks if a tensor has the expected number of dimensions.
     *
     * @param tensor The tensor to check.
     * @param expectedDims The expected number of dimensions.
     * @param operationName The name of the operation being performed.
     * @throws IllegalArgumentException If the number of dimensions doesn't match.
     */
    fun checkDimensions(tensor: Tensor, expectedDims: Int, operationName: String) {
        if (tensor.shape.rank != expectedDims) {
            throw IllegalArgumentException(
                "Dimension mismatch for $operationName operation: " +
                "tensor has ${tensor.shape.rank} dimensions, " +
                "expected $expectedDims dimensions"
            )
        }
    }

    /**
     * Checks if a tensor has at least the specified number of dimensions.
     *
     * @param tensor The tensor to check.
     * @param minDims The minimum number of dimensions.
     * @param operationName The name of the operation being performed.
     * @throws IllegalArgumentException If the number of dimensions is less than expected.
     */
    fun checkMinDimensions(tensor: Tensor, minDims: Int, operationName: String) {
        if (tensor.shape.rank < minDims) {
            throw IllegalArgumentException(
                "Dimension mismatch for $operationName operation: " +
                "tensor has ${tensor.shape.rank} dimensions, " +
                "expected at least $minDims dimensions"
            )
        }
    }

    /**
     * Computes the result shape for an element-wise operation.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @param operationName The name of the operation being performed.
     * @return The shape of the result tensor.
     * @throws IllegalArgumentException If the shapes are not compatible.
     */
    fun computeElementWiseShape(left: Tensor, right: Tensor, operationName: String): Shape {
        checkSameShape(left, right, operationName)
        return left.shape
    }

    /**
     * Computes the result shape for a matrix multiplication operation.
     *
     * @param left The left tensor.
     * @param right The right tensor.
     * @return The shape of the result tensor.
     * @throws IllegalArgumentException If the tensors are not compatible.
     */
    fun computeMatmulShape(left: Tensor, right: Tensor): Shape {
        checkMatmulCompatible(left, right)
        return Shape(left.shape[0], right.shape[1])
    }
}
