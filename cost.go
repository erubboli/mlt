package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func Cost(x *mat64.Dense, y, theta *mat64.Vector) float64 {
	m, _ := x.Dims()
	_, n := theta.Dims()
	h := mat64.NewDense(m, n, make([]float64, m*n))
	squaredErrors := mat64.NewDense(m, 1, make([]float64, m))

	h.Mul(x, theta)
	squaredErrors.Apply(func(r, c int, v float64) float64 {
		return math.Pow(h.At(r, c)-y.At(r, c), 2)
	}, squaredErrors)

	l := 1.0 / (2.0 * float64(m))
	j := mat64.Sum(squaredErrors) * l
	return j
}
