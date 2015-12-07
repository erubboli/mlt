//function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
package main

import "github.com/gonum/matrix/mat64"

func GradientDescent(X *mat64.Dense, y, theta *mat64.Vector, alpha float64, numIters int) *mat64.Vector {
	m := y.Len()     // Number Of Training Examples
	n := theta.Len() // Number Of Features
	h := *mat64.NewDense(m, 1, nil)
	delta := mat64.NewDense(n, 1, nil)
	new_theta := mat64.NewDense(n, 1, nil)
	d := float64(1.0 / float64(m))

	for i := 0; i < numIters; i++ {
		h.Mul(X, new_theta)
		h.Apply(func(r, c int, v float64) float64 {
			return (v - y.At(r, c)) * d
		}, &h)
		delta.Mul(X.T(), &h)
		new_theta.Apply(func(r, c int, v float64) float64 {
			return v - (alpha * delta.At(r, c))
		}, new_theta)
	}
	return new_theta.ColView(0)
}
