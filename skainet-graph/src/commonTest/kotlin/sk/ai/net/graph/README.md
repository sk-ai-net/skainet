# Unit Testing Coverage

This document provides an overview of the unit testing coverage for the Composable Compute Graphs project.

## Current Test Coverage Summary

| Module | Components Tested | Components Needing Tests | Coverage Status |
|--------|------------------|-------------------------|----------------|
| Core   | 4/4 (100%)       | 0                       | ✅ Complete     |
| Tensor | 2/? (~50%)       | Several tensor types    | ⚠️ Partial     |
| Neural Network | 0/6 (0%) | All components          | ❌ Missing      |
| Autodiff | 0/6 (0%)       | All components          | ❌ Missing      |
| Memory Management | 0/3 (0%) | All components       | ❌ Missing      |
| Backend | 0/3 (0%)        | All components          | ❌ Missing      |
| Optimization | 0/3 (0%)    | All components         | ❌ Missing      |
| Serialization | 0/3 (0%)   | All components         | ❌ Missing      |
| **Overall** | **6/~30 (~20%)** | **Most components** | ⚠️ **Needs Improvement** |

### Core Components (✅ Complete)
- [x] ComputeNode (ComputeNodeTest.kt)
  - Tests for ValueNode, AddNode, MultiplyNode, ActivationNode
  - Tests for complex graph construction and evaluation
- [x] Expression (ExpressionTest.kt)
  - Tests for value expressions, addition, multiplication
  - Tests for complex expression building
  - Tests for different data types (Int, String, Double)
- [x] LazyComputeNode (LazyComputeNodeTest.kt)
  - Tests for lazy evaluation and caching
  - Tests for cache clearing
  - Tests for extension function
  - Tests for complex graph with lazy evaluation
- [x] dumpGraph (DumpGraphTest.kt)
  - Tests for printing different node types
  - Tests for printing complex graphs
  - Tests for custom indentation
  - Tests for handling unknown node types

### Tensor Components (⚠️ Partial)
- [x] TernaryTensor (TernaryTensorTest.kt)
  - Tests for creation and access
  - Tests for large random data
  - Tests for memory usage
- [x] MixedTensor (MixedTensorTest.kt)
  - Tests for operations between different tensor types
  - Tests for matrix multiplication
  - Tests for neural network with mixed tensors

## Components Needing Tests

### Neural Network Components (❌ Missing)
- [ ] Module (Module.kt)
- [ ] Linear (Linear.kt)
- [ ] DifferentiableLinear (DifferentiableLinear.kt)
- [ ] ReLU (ReLU.kt)
- [ ] DifferentiableActivation (DifferentiableActivation.kt)
- [ ] Sequential (Sequential.kt)

### Automatic Differentiation (❌ Missing)
- [ ] Differentiable (Differentiable.kt)
- [ ] DifferentiableComputeNode (DifferentiableComputeNode.kt)
- [ ] DifferentiableAddNode (DifferentiableAddNode.kt)
- [ ] DifferentiableMultiplyNode (DifferentiableMultiplyNode.kt)
- [ ] DifferentiableActivationNode (DifferentiableActivationNode.kt)
- [ ] GradientRegistry (GradientRegistry.kt)

### Memory Management (❌ Missing)
- [ ] Memory allocation and deallocation
- [ ] Memory pooling
- [ ] Out-of-core computation (DiskBackedTensor)

### Backend Implementations (❌ Missing)
- [ ] CPU backend
- [ ] GPU backend (if applicable)
- [ ] Platform-specific optimizations

### Optimization Algorithms (❌ Missing)
- [ ] GraphOptimizer (GraphOptimizer.kt)
- [ ] Operation fusion
- [ ] Lazy evaluation strategies

### Serialization (❌ Missing)
- [ ] Model serialization
- [ ] Model deserialization
- [ ] Cross-platform compatibility

## Testing Approach

The testing approach for this project follows these principles:

1. **Comprehensive Coverage**: Each component should have tests that verify its functionality.
2. **Cross-Platform Testing**: Tests should run on all supported platforms (desktop, iOS, WebAssembly).
3. **Edge Cases**: Tests should include edge cases and error conditions.
4. **Performance Testing**: Critical components should have performance tests.

## Running Tests

To run all tests:
```
./gradlew allTests
```

To run tests for a specific component:
```
./gradlew :skainet-graph:jvmTest --tests "sk.ai.net.graph.core.ComputeNodeTest"
```

## Code Coverage Tools

The project uses Kover for code coverage analysis, but it's currently only enabled for the composeApp module. To enable it for the skainet-graph module:

1. Uncomment the Kover plugin in skainet-graph/build.gradle.kts:
   ```kotlin
   // Change this:
   //alias(libs.plugins.kover)
   // To this:
   alias(libs.plugins.kover)
   ```

2. Run the Kover report:
   ```
   ./gradlew :skainet-graph:koverHtmlReport
   ```

3. View the report at:
   ```
   skainet-graph/build/reports/kover/html/index.html
   ```

## Recommendations for Improving Test Coverage

1. **Prioritize Neural Network and Autodiff Components**: These are critical for the core functionality of the project.
2. **Create Test Templates**: Use existing tests as templates for new components.
3. **Test Cross-Platform Behavior**: Ensure tests run on all supported platforms.
4. **Add Integration Tests**: Test how components work together in real-world scenarios.
5. **Set Up CI/CD**: Configure continuous integration to run tests automatically.
6. **Track Coverage Metrics**: Use Kover to track coverage metrics over time.

## Contributing Tests

When adding new components to the project, please also add corresponding tests. Follow the existing test patterns and ensure that tests are comprehensive and cross-platform compatible.
