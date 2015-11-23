package main

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func Cost(x *mat64.Dense, y, theta *mat64.Vector) float64 {
	//initialize receivers
	m, _ := x.Dims()
	h := mat64.NewDense(m, 1, make([]float64, m))
	squaredErrors := mat64.NewDense(m, 1, make([]float64, m))

	//actual calculus
	h.Mul(x, theta)
	squaredErrors.Apply(func(r, c int, v float64) float64 {
		return math.Pow(h.At(r, c)-y.At(r, c), 2)
	}, h)
	j := mat64.Sum(squaredErrors) * 1.0 / (2.0 * float64(m))

	return j
}
