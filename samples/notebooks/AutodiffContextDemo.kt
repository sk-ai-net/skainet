package sk.ai.net.samples.notebooks

import sk.ai.net.graph.tensor.shape.Shape
import sk.ai.net.graph.tensor.Tensor
import sk.ai.net.autograd.AutodiffContext
import sk.ai.net.autograd.AutodiffMode
import sk.ai.net.autograd.AutogradTensor
import sk.ai.net.autograd.withAutodiff
import sk.ai.net.autograd.requireGradient
import sk.ai.net.impl.DoublesTensor

/**
 * This sample demonstrates the usage of the AutodiffContext for automatic differentiation.
 * It shows how to switch between training and inference modes, and how to perform tensor operations
 * with the new context-based approach.
 */
fun main() {
    println("AutodiffContext Demo")
    println("====================")
    
    // Example 1: Creating tensors in training mode
    println("\nExample 1: Creating tensors in training mode")
    println("--------------------------------------------")
    
    AutodiffContext.training {
        // Create a tensor in training mode with requiresGrad = true
        val x = AutodiffContext.current().tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)
        println("Tensor x: $x")
        println("x is AutogradTensor: ${x is AutogradTensor}")
        println("x requires gradients: ${(x as? AutogradTensor)?.requiresGrad}")
        
        // Create a tensor in training mode with requiresGrad = false
        val y = AutodiffContext.current().tensor(Shape(2, 2), doubleArrayOf(2.0, 3.0, 4.0, 5.0), requiresGrad = false)
        println("Tensor y: $y")
        println("y is AutogradTensor: ${y is AutogradTensor}")
        println("y requires gradients: ${(y as? AutogradTensor)?.requiresGrad}")
        
        // Perform operations
        val z = x.plus(y)
        println("z = x + y: $z")
        println("z is AutogradTensor: ${z is AutogradTensor}")
        println("z requires gradients: ${(z as? AutogradTensor)?.requiresGrad}")
        
        // Compute gradients
        if (z is AutogradTensor) {
            z.backward()
            println("Gradient of x: ${(x as AutogradTensor).grad}")
            println("Gradient of y: ${(y as AutogradTensor).grad}")
        }
    }
    
    // Example 2: Creating tensors in inference mode
    println("\nExample 2: Creating tensors in inference mode")
    println("--------------------------------------------")
    
    AutodiffContext.inference {
        // Create a tensor in inference mode
        val x = AutodiffContext.current().tensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0), requiresGrad = true)
        println("Tensor x: $x")
        println("x is AutogradTensor: ${x is AutogradTensor}")
        
        // Create another tensor in inference mode
        val y = AutodiffContext.current().tensor(Shape(2, 2), doubleArrayOf(2.0, 3.0, 4.0, 5.0), requiresGrad = false)
        println("Tensor y: $y")
        println("y is AutogradTensor: ${y is AutogradTensor}")
        
        // Perform operations
        val z = x.plus(y)
        println("z = x + y: $z")
        println("z is AutogradTensor: ${z is AutogradTensor}")
    }
    
    // Example 3: Switching between modes
    println("\nExample 3: Switching between modes")
    println("--------------------------------")
    
    // Create tensors in training mode
    val tensorsInTraining = AutodiffContext.training {
        listOf(
            AutodiffContext.current().tensor(Shape(2, 2), 1.0, requiresGrad = true),
            AutodiffContext.current().tensor(Shape(2, 2), 2.0, requiresGrad = false)
        )
    }
    
    println("Tensors created in training mode:")
    println("tensor[0] is AutogradTensor: ${tensorsInTraining[0] is AutogradTensor}")
    println("tensor[1] is AutogradTensor: ${tensorsInTraining[1] is AutogradTensor}")
    
    // Create tensors in inference mode
    val tensorsInInference = AutodiffContext.inference {
        listOf(
            AutodiffContext.current().tensor(Shape(2, 2), 1.0, requiresGrad = true),
            AutodiffContext.current().tensor(Shape(2, 2), 2.0, requiresGrad = false)
        )
    }
    
    println("\nTensors created in inference mode:")
    println("tensor[0] is AutogradTensor: ${tensorsInInference[0] is AutogradTensor}")
    println("tensor[1] is AutogradTensor: ${tensorsInInference[1] is AutogradTensor}")
    
    // Example 4: Using extension functions
    println("\nExample 4: Using extension functions")
    println("----------------------------------")
    
    // Create a regular DoublesTensor
    val regularTensor = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 2.0, 3.0, 4.0))
    println("Regular tensor: $regularTensor")
    
    // Use withAutodiff in training mode
    AutodiffContext.training {
        val autogradTensor = regularTensor.withAutodiff(true)
        println("\nIn training mode:")
        println("autogradTensor is AutogradTensor: ${autogradTensor is AutogradTensor}")
        println("autogradTensor requires gradients: ${(autogradTensor as? AutogradTensor)?.requiresGrad}")
        
        // Use requireGradient in training mode
        val gradTensor = regularTensor.requireGradient()
        println("gradTensor is AutogradTensor: ${gradTensor is AutogradTensor}")
        println("gradTensor requires gradients: ${(gradTensor as? AutogradTensor)?.requiresGrad}")
    }
    
    // Use withAutodiff in inference mode
    AutodiffContext.inference {
        val inferTensor = regularTensor.withAutodiff(true)
        println("\nIn inference mode:")
        println("inferTensor is AutogradTensor: ${inferTensor is AutogradTensor}")
        
        // Use requireGradient in inference mode
        val noGradTensor = regularTensor.requireGradient()
        println("noGradTensor is AutogradTensor: ${noGradTensor is AutogradTensor}")
    }
    
    // Example 5: Complex operations and gradient computation
    println("\nExample 5: Complex operations and gradient computation")
    println("-------------------------------------------------")
    
    AutodiffContext.training {
        // Create tensors
        val a = AutodiffContext.current().tensor(Shape(2, 3), doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), requiresGrad = true)
        val b = AutodiffContext.current().tensor(Shape(3, 2), doubleArrayOf(7.0, 8.0, 9.0, 10.0, 11.0, 12.0), requiresGrad = true)
        
        // Perform matrix multiplication
        val c = a.matmul(b)
        println("a: $a")
        println("b: $b")
        println("c = a.matmul(b): $c")
        
        // Compute gradients
        if (c is AutogradTensor) {
            // Create a gradient for c
            val gradC = DoublesTensor(Shape(2, 2), doubleArrayOf(1.0, 1.0, 1.0, 1.0))
            c.backward(gradC)
            
            println("\nGradients:")
            println("Gradient of a: ${(a as AutogradTensor).grad}")
            println("Gradient of b: ${(b as AutogradTensor).grad}")
        }
    }
    
    // Example 6: Default requiresGrad
    println("\nExample 6: Default requiresGrad")
    println("-----------------------------")
    
    // Create a context with defaultRequiresGrad = true
    AutodiffContext.withContext(AutodiffMode.TRAINING, defaultRequiresGrad = true) {
        // Create a tensor without specifying requiresGrad
        val x = AutodiffContext.current().tensor(Shape(2, 2), 1.0)
        println("With defaultRequiresGrad = true:")
        println("x is AutogradTensor: ${x is AutogradTensor}")
        println("x requires gradients: ${(x as? AutogradTensor)?.requiresGrad}")
    }
    
    // Create a context with defaultRequiresGrad = false
    AutodiffContext.withContext(AutodiffMode.TRAINING, defaultRequiresGrad = false) {
        // Create a tensor without specifying requiresGrad
        val x = AutodiffContext.current().tensor(Shape(2, 2), 1.0)
        println("\nWith defaultRequiresGrad = false:")
        println("x is AutogradTensor: ${x is AutogradTensor}")
        println("x requires gradients: ${(x as? AutogradTensor)?.requiresGrad}")
    }
}