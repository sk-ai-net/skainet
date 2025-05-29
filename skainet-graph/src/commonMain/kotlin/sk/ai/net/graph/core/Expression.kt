package sk.ai.net.graph.core

/**
 * A wrapper class for building computation graphs using a natural expression syntax.
 *
 * This class wraps a ComputeNode and provides extension functions for building
 * complex expressions using operator overloading.
 *
 * @param T The type of data processed by the computation node.
 * @property node The wrapped computation node.
 */
class Expr<T>(val node: ComputeNode<T>)

/**
 * Creates an expression from a value.
 *
 * This function creates a ValueNode with the given value and wraps it in an Expr.
 *
 * @param T The type of the value.
 * @param value The value to wrap.
 * @return An expression containing a ValueNode with the given value.
 */
fun <T> value(value: T) = Expr(ValueNode(value))

/**
 * Adds two expressions.
 *
 * This function creates an AddNode that adds the results of the two expressions
 * using the provided addition function.
 *
 * @param T The type of the values to add.
 * @param other The expression to add to this expression.
 * @param add A function that defines how to add two values of type [T].
 * @return An expression representing the addition of the two expressions.
 */
fun <T> Expr<T>.plus(other: Expr<T>, add: (T, T) -> T) =
    Expr(AddNode(add).apply {
        inputs += this@plus.node
        inputs += other.node
    })

/**
 * Multiplies two expressions.
 *
 * This function creates a MultiplyNode that multiplies the results of the two expressions
 * using the provided multiplication function.
 *
 * @param T The type of the values to multiply.
 * @param other The expression to multiply with this expression.
 * @param multiply A function that defines how to multiply two values of type [T].
 * @return An expression representing the multiplication of the two expressions.
 */
fun <T> Expr<T>.times(other: Expr<T>, multiply: (T, T) -> T) =
    Expr(MultiplyNode(multiply).apply {
        inputs += this@times.node
        inputs += other.node
    })
