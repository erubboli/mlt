package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type LinearRegression struct {
	Theta     *mat64.Vector
	alpha     float64
	tolerance float64
	maxIters  int
	x         *mat64.Dense
	y         *mat64.Vector
	n         int
	m         int
}

func NewLinearRegression(x *mat64.Dense, y *mat64.Vector) *LinearRegression {
	m, n := x.Dims()
	theta := mat64.NewVector(n, nil)
	return &LinearRegression{x: x, y: y, alpha: 0.001, tolerance: 0.00001, maxIters: 10.000, n: n, m: m, Theta: theta}

}

func (lr *LinearRegression) Fit() {
	h := *mat64.NewVector(lr.m, nil)
	partials := mat64.NewVector(lr.n, nil)
	alpha_m := lr.alpha / float64(lr.m)

Descent:
	for i := 0; i < lr.maxIters; i++ {
		// Calculate partial derivatives
		h.MulVec(lr.x, lr.Theta)
		for x := 0; x < lr.m; x++ {
			h.SetVec(x, h.At(x, 0)-lr.y.At(x, 0))
		}
		partials.MulVec(lr.x.T(), &h)

		// Update theta values with the precalculated partials
		for x := 0; x < lr.n; x++ {
			theta_j := lr.Theta.At(x, 0) - alpha_m*partials.At(x, 0)
			lr.Theta.SetVec(x, theta_j)
		}

		// Check the "distance" to the local minumum
		dist := math.Sqrt(mat64.Dot(partials, partials))

		if dist <= lr.tolerance {
			break Descent
		}
	}
}

func (lr *LinearRegression) Predict(x *mat64.Vector) float64 {
	return mat64.Dot(x, lr.Theta)
}
