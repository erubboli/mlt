//function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func GradientDescent(X *mat64.Dense, y *mat64.Vector, alpha, tolerance float64, maxIters int) *mat64.Vector {
	// m = Number of Training Examples
	// n = Number of Features
	m, n := X.Dims()
	h := mat64.NewVector(m, nil)
	partials := mat64.NewVector(n, nil)
	new_theta := mat64.NewVector(n, nil)

Regression:
	for i := 0; i < maxIters; i++ {
		// Calculate partial derivatives
		h.MulVec(X, new_theta)
		for el := 0; el < m; el++ {
			val := (h.At(el, 0) - y.At(el, 0)) / float64(m)
			h.SetVec(el, val)
		}
		partials.MulVec(X.T(), h)

		// Update theta values
		for el := 0; el < n; el++ {
			new_val := new_theta.At(el, 0) - (alpha * partials.At(el, 0))
			new_theta.SetVec(el, new_val)
		}

		// Check the "distance" to the local minumum
		dist := math.Sqrt(mat64.Dot(partials, partials))

		if dist <= tolerance {
			break Regression
		}
	}
	return new_theta
}
